"""
281. Zigzag Iterator - Multiple Approaches
Difficulty: Medium

Given two vectors of integers v1 and v2, implement an iterator to return their elements alternately.

Implement the ZigzagIterator class:
- ZigzagIterator(List<int> v1, List<int> v2) initializes the object with the two vectors v1 and v2.
- boolean hasNext() Returns true if the iterator has more elements or false.
- int next() Returns the next element of the iterator.
"""

from typing import List, Deque
from collections import deque

class ZigzagIteratorSimple:
    """
    Approach 1: Simple Two-Pointer Implementation
    
    Use two pointers to track positions in both vectors.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, v1: List[int], v2: List[int]):
        self.v1 = v1
        self.v2 = v2
        self.i1 = 0  # Pointer for v1
        self.i2 = 0  # Pointer for v2
        self.turn = 1  # 1 for v1, 2 for v2
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        if self.turn == 1:
            # Try to get from v1
            if self.i1 < len(self.v1):
                result = self.v1[self.i1]
                self.i1 += 1
                self.turn = 2
                return result
            else:
                # v1 exhausted, switch to v2
                self.turn = 2
                return self.next()
        else:
            # Try to get from v2
            if self.i2 < len(self.v2):
                result = self.v2[self.i2]
                self.i2 += 1
                self.turn = 1
                return result
            else:
                # v2 exhausted, switch to v1
                self.turn = 1
                return self.next()
    
    def hasNext(self) -> bool:
        return self.i1 < len(self.v1) or self.i2 < len(self.v2)

class ZigzagIteratorQueue:
    """
    Approach 2: Queue-based Implementation
    
    Use queue to manage iterators for clean alternation.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(k) where k is number of vectors
    """
    
    def __init__(self, v1: List[int], v2: List[int]):
        self.queue = deque()
        
        # Add non-empty vectors to queue
        if v1:
            self.queue.append((v1, 0))
        if v2:
            self.queue.append((v2, 0))
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        # Get next vector and its current index
        vector, index = self.queue.popleft()
        
        # Get element
        result = vector[index]
        
        # If more elements remain in this vector, add it back to queue
        if index + 1 < len(vector):
            self.queue.append((vector, index + 1))
        
        return result
    
    def hasNext(self) -> bool:
        return len(self.queue) > 0

class ZigzagIteratorGeneralized:
    """
    Approach 3: Generalized for Multiple Vectors
    
    Support any number of vectors with round-robin iteration.
    
    Time Complexity:
    - __init__: O(k) where k is number of vectors
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(k)
    """
    
    def __init__(self, vectors: List[List[int]]):
        self.queue = deque()
        
        # Initialize queue with all non-empty vectors
        for vector in vectors:
            if vector:
                self.queue.append((vector, 0))
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        vector, index = self.queue.popleft()
        result = vector[index]
        
        # Add back to queue if more elements remain
        if index + 1 < len(vector):
            self.queue.append((vector, index + 1))
        
        return result
    
    def hasNext(self) -> bool:
        return len(self.queue) > 0

class ZigzagIteratorAdvanced:
    """
    Approach 4: Advanced with Features and Analytics
    
    Enhanced version with statistics and additional functionality.
    
    Time Complexity:
    - __init__: O(k)
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(k + analytics)
    """
    
    def __init__(self, v1: List[int], v2: List[int]):
        self.vectors = [v1, v2]
        self.queue = deque()
        
        # Initialize with non-empty vectors
        for i, vector in enumerate(self.vectors):
            if vector:
                self.queue.append((i, 0))  # (vector_index, element_index)
        
        # Analytics
        self.total_elements = len(v1) + len(v2)
        self.elements_consumed = 0
        self.next_calls = 0
        self.has_next_calls = 0
        
        # Features
        self.history = []  # Track iteration pattern
        self.vector_stats = [0, 0]  # Elements consumed from each vector
    
    def next(self) -> int:
        self.next_calls += 1
        
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        vector_idx, element_idx = self.queue.popleft()
        vector = self.vectors[vector_idx]
        
        result = vector[element_idx]
        
        # Update analytics
        self.elements_consumed += 1
        self.vector_stats[vector_idx] += 1
        self.history.append((vector_idx, element_idx, result))
        
        # Add back to queue if more elements remain
        if element_idx + 1 < len(vector):
            self.queue.append((vector_idx, element_idx + 1))
        
        return result
    
    def hasNext(self) -> bool:
        self.has_next_calls += 1
        return len(self.queue) > 0
    
    def get_statistics(self) -> dict:
        """Get iteration statistics"""
        progress_percent = (self.elements_consumed / max(1, self.total_elements)) * 100
        
        return {
            'total_elements': self.total_elements,
            'elements_consumed': self.elements_consumed,
            'progress_percent': progress_percent,
            'next_calls': self.next_calls,
            'has_next_calls': self.has_next_calls,
            'vector_stats': self.vector_stats.copy(),
            'remaining_in_queue': len(self.queue)
        }
    
    def get_pattern(self) -> List[int]:
        """Get the pattern of vector access (0 for v1, 1 for v2)"""
        return [vector_idx for vector_idx, _, _ in self.history]
    
    def peek_next_source(self) -> int:
        """Peek at which vector the next element comes from"""
        if not self.hasNext():
            return -1
        
        vector_idx, _ = self.queue[0]
        return vector_idx
    
    def reset(self) -> None:
        """Reset iterator to beginning"""
        self.queue.clear()
        
        # Re-initialize queue
        for i, vector in enumerate(self.vectors):
            if vector:
                self.queue.append((i, 0))
        
        # Reset analytics
        self.elements_consumed = 0
        self.next_calls = 0
        self.has_next_calls = 0
        self.history.clear()
        self.vector_stats = [0, 0]

class ZigzagIteratorMemoryOptimized:
    """
    Approach 5: Memory-Optimized Implementation
    
    Minimize memory overhead for large vectors.
    
    Time Complexity:
    - __init__: O(1)
    - next: O(1)
    - hasNext: O(1)
    
    Space Complexity: O(1)
    """
    
    def __init__(self, v1: List[int], v2: List[int]):
        self.v1 = v1
        self.v2 = v2
        self.state = self._initialize_state()
    
    def _initialize_state(self) -> tuple:
        """Initialize state for iteration"""
        if self.v1 and self.v2:
            return (0, 0, 0)  # (i1, i2, current_vector: 0=v1, 1=v2)
        elif self.v1:
            return (0, 0, 0)  # Only v1 available
        elif self.v2:
            return (0, 0, 1)  # Only v2 available
        else:
            return (0, 0, -1)  # No vectors available
    
    def next(self) -> int:
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        i1, i2, current = self.state
        
        if current == 0:  # Get from v1
            result = self.v1[i1]
            i1 += 1
            
            # Switch to v2 if available and has elements
            if i2 < len(self.v2):
                current = 1
            # If v2 exhausted but v1 has more, stay with v1
            
            self.state = (i1, i2, current)
            return result
        
        else:  # Get from v2
            result = self.v2[i2]
            i2 += 1
            
            # Switch to v1 if available and has elements
            if i1 < len(self.v1):
                current = 0
            # If v1 exhausted but v2 has more, stay with v2
            
            self.state = (i1, i2, current)
            return result
    
    def hasNext(self) -> bool:
        i1, i2, _ = self.state
        return i1 < len(self.v1) or i2 < len(self.v2)


def test_zigzag_iterator_basic():
    """Test basic zigzag iterator functionality"""
    print("=== Testing Basic Zigzag Iterator Functionality ===")
    
    implementations = [
        ("Simple Two Pointer", ZigzagIteratorSimple),
        ("Queue Based", ZigzagIteratorQueue),
        ("Advanced", ZigzagIteratorAdvanced),
        ("Memory Optimized", ZigzagIteratorMemoryOptimized)
    ]
    
    test_cases = [
        ([1, 2], [3, 4, 5, 6]),
        ([1], []),
        ([], [2, 3]),
        ([1, 2, 3], [4, 5]),
        ([], [])
    ]
    
    for v1, v2 in test_cases:
        print(f"\nTest case: v1={v1}, v2={v2}")
        
        # Generate expected output
        expected = []
        i1 = i2 = 0
        turn = 0  # 0 for v1, 1 for v2
        
        while i1 < len(v1) or i2 < len(v2):
            if turn == 0:
                if i1 < len(v1):
                    expected.append(v1[i1])
                    i1 += 1
                turn = 1
            else:
                if i2 < len(v2):
                    expected.append(v2[i2])
                    i2 += 1
                turn = 0
        
        for name, IteratorClass in implementations:
            try:
                iterator = IteratorClass(v1, v2)
                result = []
                
                while iterator.hasNext():
                    result.append(iterator.next())
                
                correct = result == expected
                print(f"  {name}: {result} - {'✓' if correct else '✗'}")
                
            except Exception as e:
                print(f"  {name}: Error - {e}")

def test_zigzag_iterator_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Zigzag Iterator Edge Cases ===")
    
    # Test both vectors empty
    print("Both vectors empty:")
    iterator = ZigzagIteratorQueue([], [])
    
    print(f"  hasNext(): {iterator.hasNext()}")
    
    try:
        iterator.next()
        print("  next(): Should have raised exception")
    except StopIteration:
        print("  next(): Correctly raised StopIteration")
    
    # Test unequal length vectors
    print(f"\nUnequal length vectors:")
    iterator = ZigzagIteratorAdvanced([1, 2, 3, 4, 5], [10, 20])
    
    result = []
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"  Result: {result}")
    print(f"  Expected: [1, 10, 2, 20, 3, 4, 5]")
    
    # Test single element vectors
    print(f"\nSingle element vectors:")
    iterator = ZigzagIteratorSimple([100], [200])
    
    result = []
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"  Result: {result}")
    print(f"  Expected: [100, 200]")
    
    # Test one empty vector
    print(f"\nOne empty vector:")
    iterator = ZigzagIteratorMemoryOptimized([1, 2, 3], [])
    
    result = []
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"  Result: {result}")
    print(f"  Expected: [1, 2, 3]")

def test_generalized_iterator():
    """Test generalized iterator for multiple vectors"""
    print("\n=== Testing Generalized Iterator ===")
    
    # Test with multiple vectors
    vectors = [
        [1, 4, 7],
        [2, 5, 8, 9],
        [3, 6]
    ]
    
    iterator = ZigzagIteratorGeneralized(vectors)
    
    result = []
    while iterator.hasNext():
        result.append(iterator.next())
    
    print(f"Vectors: {vectors}")
    print(f"Result: {result}")
    print(f"Expected pattern: 1,2,3,4,5,6,7,8,9")
    
    # Test with some empty vectors
    vectors_with_empty = [
        [1, 3],
        [],
        [2, 4, 5],
        []
    ]
    
    iterator2 = ZigzagIteratorGeneralized(vectors_with_empty)
    
    result2 = []
    while iterator2.hasNext():
        result2.append(iterator2.next())
    
    print(f"\nVectors with empty: {vectors_with_empty}")
    print(f"Result: {result2}")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    iterator = ZigzagIteratorAdvanced([1, 3, 5, 7], [2, 4, 6])
    
    # Test iteration with statistics
    print("Iteration with statistics:")
    
    for i in range(4):
        if iterator.hasNext():
            element = iterator.next()
            stats = iterator.get_statistics()
            
            print(f"  Element {i+1}: {element}")
            print(f"    Progress: {stats['progress_percent']:.1f}%")
            print(f"    From vector: {iterator.peek_next_source()}")
    
    # Test pattern analysis
    pattern = iterator.get_pattern()
    print(f"\nAccess pattern so far: {pattern}")
    print("  (0 = first vector, 1 = second vector)")
    
    # Continue iteration
    remaining = []
    while iterator.hasNext():
        remaining.append(iterator.next())
    
    print(f"\nRemaining elements: {remaining}")
    
    # Final statistics
    final_stats = iterator.get_statistics()
    print(f"\nFinal statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test reset functionality
    print(f"\nTesting reset:")
    
    before_reset = iterator.get_statistics()
    iterator.reset()
    after_reset = iterator.get_statistics()
    
    print(f"  Before reset: {before_reset['elements_consumed']} consumed")
    print(f"  After reset: {after_reset['elements_consumed']} consumed")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Merging sorted datasets
    print("Application 1: Round-Robin Data Merging")
    
    dataset1 = [1, 3, 5, 7, 9]      # Odd numbers
    dataset2 = [2, 4, 6, 8, 10, 12] # Even numbers
    
    merger = ZigzagIteratorAdvanced(dataset1, dataset2)
    
    print(f"  Dataset 1 (odds): {dataset1}")
    print(f"  Dataset 2 (evens): {dataset2}")
    
    merged_data = []
    while merger.hasNext():
        merged_data.append(merger.next())
    
    print(f"  Merged result: {merged_data}")
    
    # Show access pattern
    pattern = merger.get_pattern()
    print(f"  Access pattern: {pattern}")
    
    # Application 2: Load balancing between servers
    print(f"\nApplication 2: Server Load Balancing")
    
    server1_requests = [101, 103, 105, 107]  # Server 1 request IDs
    server2_requests = [102, 104, 106]       # Server 2 request IDs
    
    load_balancer = ZigzagIteratorQueue(server1_requests, server2_requests)
    
    print("  Distributing requests round-robin:")
    print(f"    Server 1 queue: {server1_requests}")
    print(f"    Server 2 queue: {server2_requests}")
    
    request_order = []
    server_assignment = []
    current_server = 1
    
    while load_balancer.hasNext():
        request_id = load_balancer.next()
        request_order.append(request_id)
        
        # Determine which server this request came from
        if request_id in server1_requests:
            server_assignment.append(1)
        else:
            server_assignment.append(2)
    
    print(f"  Processing order: {request_order}")
    print(f"  Server assignment: {server_assignment}")
    
    # Application 3: Media playlist shuffling
    print(f"\nApplication 3: Playlist Interleaving")
    
    rock_songs = ["Rock Song 1", "Rock Song 2", "Rock Song 3"]
    pop_songs = ["Pop Song 1", "Pop Song 2", "Pop Song 3", "Pop Song 4"]
    
    # Convert to IDs for simplicity
    rock_ids = [1, 2, 3]
    pop_ids = [101, 102, 103, 104]
    
    playlist = ZigzagIteratorMemoryOptimized(rock_ids, pop_ids)
    
    print("  Creating interleaved playlist:")
    print(f"    Rock songs: {rock_songs}")
    print(f"    Pop songs: {pop_songs}")
    
    play_order = []
    song_types = []
    
    while playlist.hasNext():
        song_id = playlist.next()
        play_order.append(song_id)
        
        if song_id < 100:
            song_types.append("Rock")
        else:
            song_types.append("Pop")
    
    print(f"  Play order: {play_order}")
    print(f"  Song types: {song_types}")
    
    # Application 4: Multi-source data streaming
    print(f"\nApplication 4: Multi-Source Data Stream")
    
    source1_data = [10, 30, 50]     # Temperature readings from sensor 1
    source2_data = [20, 40, 60, 70] # Temperature readings from sensor 2
    
    stream_processor = ZigzagIteratorGeneralized([source1_data, source2_data])
    
    print("  Processing sensor data streams:")
    print(f"    Sensor 1 readings: {source1_data}")
    print(f"    Sensor 2 readings: {source2_data}")
    
    processed_readings = []
    sensor_sources = []
    
    reading_count = 0
    while stream_processor.hasNext():
        reading = stream_processor.next()
        processed_readings.append(reading)
        
        # Determine source based on reading pattern
        if reading_count % 2 == 0:
            sensor_sources.append("Sensor 1")
        else:
            sensor_sources.append("Sensor 2")
        
        reading_count += 1
    
    print(f"  Processed readings: {processed_readings}")
    print(f"  Reading sources: {sensor_sources}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    # Create large test vectors
    large_v1 = list(range(0, 10000, 2))    # Even numbers
    large_v2 = list(range(1, 10000, 2))    # Odd numbers
    
    implementations = [
        ("Simple Two Pointer", ZigzagIteratorSimple),
        ("Queue Based", ZigzagIteratorQueue),
        ("Memory Optimized", ZigzagIteratorMemoryOptimized)
    ]
    
    for name, IteratorClass in implementations:
        # Time initialization
        start_time = time.time()
        iterator = IteratorClass(large_v1, large_v2)
        init_time = (time.time() - start_time) * 1000
        
        # Time iteration
        start_time = time.time()
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        iteration_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Init: {init_time:.3f}ms")
        print(f"    Iteration: {iteration_time:.2f}ms")
        print(f"    Elements: {count}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Queue Based", ZigzagIteratorQueue),
        ("Memory Optimized", ZigzagIteratorMemoryOptimized),
        ("Advanced", ZigzagIteratorAdvanced)
    ]
    
    # Test with different vector sizes
    test_cases = [
        ("Small equal", [1, 2, 3], [4, 5, 6]),
        ("Large unequal", list(range(1000)), list(range(5000, 5100))),
        ("One empty", list(range(1000)), []),
        ("Both large", list(range(5000)), list(range(5000, 10000)))
    ]
    
    for test_name, v1, v2 in test_cases:
        print(f"\n{test_name} (v1: {len(v1)}, v2: {len(v2)}):")
        
        for impl_name, IteratorClass in implementations:
            iterator = IteratorClass(v1, v2)
            
            # Estimate memory overhead
            if hasattr(iterator, 'queue'):
                overhead = len(iterator.queue)
                approach = "Queue-based"
            elif hasattr(iterator, 'state'):
                overhead = 1  # Just state tuple
                approach = "State-based"
            else:
                overhead = 3  # Basic pointers
                approach = "Pointer-based"
            
            total_elements = len(v1) + len(v2)
            
            print(f"    {impl_name} ({approach}): ~{overhead} vs {total_elements} elements")

def stress_test_zigzag_iterator():
    """Stress test zigzag iterator"""
    print("\n=== Stress Testing Zigzag Iterator ===")
    
    import time
    
    # Create very large vectors
    large_v1 = list(range(0, 100000, 2))    # 50k even numbers
    large_v2 = list(range(1, 100000, 2))    # 50k odd numbers
    
    print(f"Stress test: {len(large_v1)} + {len(large_v2)} = {len(large_v1) + len(large_v2)} elements")
    
    iterator = ZigzagIteratorQueue(large_v1, large_v2)
    
    start_time = time.time()
    
    count = 0
    checksum = 0
    
    while iterator.hasNext():
        value = iterator.next()
        checksum += value
        count += 1
        
        # Progress update every 10k elements
        if count % 10000 == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"    Processed {count} elements, rate: {rate:.0f} elements/sec")
    
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Elements processed: {count}")
    print(f"  Checksum: {checksum}")
    print(f"  Rate: {count / total_time:.0f} elements/sec")
    
    # Verify correctness
    expected_checksum = sum(large_v1) + sum(large_v2)
    print(f"  Checksum correct: {checksum == expected_checksum}")

def test_alternation_patterns():
    """Test different alternation patterns"""
    print("\n=== Testing Alternation Patterns ===")
    
    patterns = [
        ("Equal length", [1, 3, 5], [2, 4, 6]),
        ("First longer", [1, 3, 5, 7, 9], [2, 4]),
        ("Second longer", [1, 3], [2, 4, 6, 8, 10]),
        ("First empty", [], [2, 4, 6]),
        ("Second empty", [1, 3, 5], []),
        ("Single elements", [1], [2])
    ]
    
    for pattern_name, v1, v2 in patterns:
        iterator = ZigzagIteratorAdvanced(v1, v2)
        
        result = []
        pattern_trace = []
        
        while iterator.hasNext():
            result.append(iterator.next())
        
        pattern_trace = iterator.get_pattern()
        
        print(f"  {pattern_name}:")
        print(f"    v1: {v1}, v2: {v2}")
        print(f"    Result: {result}")
        print(f"    Pattern: {pattern_trace}")

def benchmark_different_sizes():
    """Benchmark with different vector size ratios"""
    print("\n=== Benchmarking Different Size Ratios ===")
    
    import time
    
    size_ratios = [
        ("1:1", 5000, 5000),
        ("1:2", 3333, 6667),
        ("1:10", 909, 9091),
        ("10:1", 9091, 909),
        ("1:100", 99, 9901)
    ]
    
    for ratio_name, size1, size2 in size_ratios:
        v1 = list(range(size1))
        v2 = list(range(10000, 10000 + size2))
        
        iterator = ZigzagIteratorQueue(v1, v2)
        
        start_time = time.time()
        
        count = 0
        while iterator.hasNext():
            iterator.next()
            count += 1
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  Ratio {ratio_name} ({size1}:{size2}):")
        print(f"    Time: {elapsed:.2f}ms")
        print(f"    Elements: {count}")
        print(f"    Rate: {count / (elapsed / 1000):.0f} elements/sec")

if __name__ == "__main__":
    test_zigzag_iterator_basic()
    test_zigzag_iterator_edge_cases()
    test_generalized_iterator()
    test_advanced_features()
    demonstrate_applications()
    test_performance_comparison()
    test_memory_efficiency()
    stress_test_zigzag_iterator()
    test_alternation_patterns()
    benchmark_different_sizes()

"""
Zigzag Iterator Design demonstrates key concepts:

Core Approaches:
1. Simple Two Pointer - Track positions in both vectors with turn indicator
2. Queue Based - Use queue to manage vector iterators for clean alternation
3. Generalized - Extend to support any number of vectors with round-robin
4. Advanced - Enhanced with analytics, pattern tracking, and features
5. Memory Optimized - Minimal overhead using compact state representation

Key Design Principles:
- Round-robin iteration across multiple data sources
- Graceful handling of unequal-length inputs
- Queue-based state management for extensibility
- Iterator pattern with hasNext/next interface

Performance Characteristics:
- All approaches: O(1) time per operation
- Simple: O(1) space overhead
- Queue: O(k) space where k is number of active vectors
- Memory optimized: Minimal constant space

Real-world Applications:
- Round-robin data merging from multiple sources
- Load balancing requests across server queues
- Media playlist interleaving for variety
- Multi-source data stream processing
- Fair scheduling algorithms
- Database result set merging

The queue-based approach provides the best balance
of simplicity, extensibility, and performance,
making it easy to extend to multiple data sources
while maintaining clean alternation logic.
"""
