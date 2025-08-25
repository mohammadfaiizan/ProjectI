"""
900. RLE Iterator - Multiple Approaches
Difficulty: Medium

We can use run-length encoding (a form of data compression) to encode a sequence of integers. In a run-length encoded array, we can represent the original array as a 2-element array encoding for each run of the sequence. For example, the array [8,8,8,1,1,1,1,1,2,1,1,1] can be represented as encoding = [3,8,5,1,1,2,3,1].

Design an iterator that iterates through a run-length encoded sequence.
- RLEIterator(int[] encoding) Initializes the object with the encoded array encoding.
- int next(int n) Exhausts the next n elements and returns the last element exhausted this way. If there is no element left to exhaust, return -1.
"""

from typing import List

class RLEIteratorSimple:
    """
    Approach 1: Simple Linear Scan
    
    Track current position and remaining count in current run.
    
    Time Complexity: 
    - __init__: O(1)
    - next: O(k) where k is number of runs to skip
    
    Space Complexity: O(1) additional space
    """
    
    def __init__(self, encoding: List[int]):
        self.encoding = encoding
        self.index = 0  # Current run index (pairs of count, value)
        self.current_count = 0  # Remaining in current run
    
    def next(self, n: int) -> int:
        while n > 0 and self.index < len(self.encoding):
            # Get current run info
            if self.current_count == 0:
                # Start new run
                if self.index + 1 < len(self.encoding):
                    self.current_count = self.encoding[self.index]
                    self.index += 2  # Move to next run
                else:
                    return -1
            
            # Calculate how many we can take from current run
            take = min(n, self.current_count)
            self.current_count -= take
            n -= take
            
            if n == 0:
                # We've taken exactly what we need
                return self.encoding[self.index - 1]  # Return value of current run
        
        return -1

class RLEIteratorOptimized:
    """
    Approach 2: Optimized with Preprocessing
    
    Preprocess to cumulative counts for faster skipping.
    
    Time Complexity: 
    - __init__: O(n) where n is number of runs
    - next: O(log n) with binary search
    
    Space Complexity: O(n)
    """
    
    def __init__(self, encoding: List[int]):
        self.runs = []  # List of (cumulative_count, value)
        self.total_consumed = 0
        
        cumulative = 0
        for i in range(0, len(encoding), 2):
            if i + 1 < len(encoding):
                count = encoding[i]
                value = encoding[i + 1]
                cumulative += count
                self.runs.append((cumulative, value))
    
    def next(self, n: int) -> int:
        self.total_consumed += n
        
        # Binary search for the run containing total_consumed
        left, right = 0, len(self.runs) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if self.runs[mid][0] >= self.total_consumed:
                if mid == 0 or self.runs[mid - 1][0] < self.total_consumed:
                    return self.runs[mid][1]
                right = mid - 1
            else:
                left = mid + 1
        
        return -1

class RLEIteratorLazy:
    """
    Approach 3: Lazy Evaluation with Generator
    
    Use generator to lazily produce elements.
    
    Time Complexity: 
    - __init__: O(1)
    - next: O(n) to consume n elements
    
    Space Complexity: O(1)
    """
    
    def __init__(self, encoding: List[int]):
        self.encoding = encoding
        self.generator = self._generate_elements()
        self.exhausted = False
    
    def _generate_elements(self):
        """Generator that yields elements from RLE encoding"""
        for i in range(0, len(self.encoding), 2):
            if i + 1 < len(self.encoding):
                count = self.encoding[i]
                value = self.encoding[i + 1]
                
                for _ in range(count):
                    yield value
    
    def next(self, n: int) -> int:
        if self.exhausted:
            return -1
        
        last_value = -1
        
        try:
            for _ in range(n):
                last_value = next(self.generator)
        except StopIteration:
            self.exhausted = True
            if last_value == -1:
                return -1
        
        return last_value

class RLEIteratorQueue:
    """
    Approach 4: Queue-based Processing
    
    Use queue to process runs efficiently.
    
    Time Complexity: 
    - __init__: O(n) to populate queue
    - next: O(k) where k is runs to process
    
    Space Complexity: O(n)
    """
    
    def __init__(self, encoding: List[int]):
        from collections import deque
        
        self.queue = deque()
        
        # Populate queue with (count, value) pairs
        for i in range(0, len(encoding), 2):
            if i + 1 < len(encoding):
                count = encoding[i]
                value = encoding[i + 1]
                if count > 0:  # Only add non-empty runs
                    self.queue.append([count, value])
    
    def next(self, n: int) -> int:
        while n > 0 and self.queue:
            count, value = self.queue[0]
            
            if count <= n:
                # Consume entire current run
                n -= count
                self.queue.popleft()
                
                if n == 0:
                    return value
            else:
                # Partially consume current run
                self.queue[0][0] -= n
                return value
        
        return -1

class RLEIteratorAdvanced:
    """
    Approach 5: Advanced with Statistics and Buffering
    
    Enhanced version with operation tracking and intelligent buffering.
    
    Time Complexity: 
    - __init__: O(n)
    - next: O(log n) average case
    
    Space Complexity: O(n)
    """
    
    def __init__(self, encoding: List[int]):
        self.original_encoding = encoding[:]
        self.runs = []
        self.total_elements = 0
        self.consumed = 0
        self.next_calls = 0
        
        # Process encoding into runs
        cumulative = 0
        for i in range(0, len(encoding), 2):
            if i + 1 < len(encoding):
                count = encoding[i]
                value = encoding[i + 1]
                
                if count > 0:
                    start_pos = cumulative
                    end_pos = cumulative + count - 1
                    
                    self.runs.append({
                        'start': start_pos,
                        'end': end_pos,
                        'count': count,
                        'value': value
                    })
                    
                    cumulative += count
        
        self.total_elements = cumulative
    
    def next(self, n: int) -> int:
        self.next_calls += 1
        
        if self.consumed + n > self.total_elements:
            # Not enough elements remaining
            self.consumed = self.total_elements
            return -1
        
        # Update consumed count
        self.consumed += n
        target_position = self.consumed - 1  # 0-indexed position of last consumed
        
        # Binary search for the run containing target position
        left, right = 0, len(self.runs) - 1
        
        while left <= right:
            mid = (left + right) // 2
            run = self.runs[mid]
            
            if run['start'] <= target_position <= run['end']:
                return run['value']
            elif target_position < run['start']:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
    
    def get_statistics(self) -> dict:
        """Get iterator statistics"""
        remaining = self.total_elements - self.consumed
        
        return {
            'total_elements': self.total_elements,
            'consumed': self.consumed,
            'remaining': remaining,
            'next_calls': self.next_calls,
            'total_runs': len(self.runs),
            'progress_percentage': (self.consumed / max(1, self.total_elements)) * 100
        }
    
    def get_current_run(self) -> dict:
        """Get information about current run"""
        if self.consumed == 0:
            return {'run_index': 0, 'position_in_run': 0}
        
        target_position = self.consumed - 1
        
        for i, run in enumerate(self.runs):
            if run['start'] <= target_position <= run['end']:
                position_in_run = target_position - run['start']
                return {
                    'run_index': i,
                    'position_in_run': position_in_run,
                    'run_value': run['value'],
                    'run_total_count': run['count'],
                    'run_remaining': run['count'] - position_in_run - 1
                }
        
        return {'run_index': -1, 'position_in_run': -1}
    
    def peek_ahead(self, n: int) -> List[int]:
        """Peek at next n elements without consuming them"""
        if self.consumed + n > self.total_elements:
            n = self.total_elements - self.consumed
        
        result = []
        
        for i in range(n):
            target_position = self.consumed + i
            
            # Find run containing this position
            for run in self.runs:
                if run['start'] <= target_position <= run['end']:
                    result.append(run['value'])
                    break
        
        return result


def test_rle_iterator_basic():
    """Test basic RLE iterator functionality"""
    print("=== Testing Basic RLE Iterator Functionality ===")
    
    implementations = [
        ("Simple Linear", RLEIteratorSimple),
        ("Optimized Binary Search", RLEIteratorOptimized),
        ("Lazy Generator", RLEIteratorLazy),
        ("Queue-based", RLEIteratorQueue),
        ("Advanced", RLEIteratorAdvanced)
    ]
    
    # Test case: [3,8,5,1,1,2,3,1] represents [8,8,8,1,1,1,1,1,2,1,1,1]
    encoding = [3, 8, 5, 1, 1, 2, 3, 1]
    
    for name, IteratorClass in implementations:
        print(f"\n{name}:")
        
        iterator = IteratorClass(encoding)
        
        # Test sequence from problem description
        test_calls = [2, 1, 1, 2]
        
        for i, n in enumerate(test_calls):
            result = iterator.next(n)
            print(f"  next({n}): {result}")

def test_rle_iterator_edge_cases():
    """Test RLE iterator edge cases"""
    print("\n=== Testing RLE Iterator Edge Cases ===")
    
    # Empty encoding
    print("Empty encoding:")
    iterator = RLEIteratorSimple([])
    result = iterator.next(1)
    print(f"  next(1): {result}")
    
    # Single run
    print(f"\nSingle run [5, 42]:")
    iterator = RLEIteratorSimple([5, 42])
    
    for i in range(7):  # Try to get more than available
        result = iterator.next(1)
        print(f"  next(1): {result}")
    
    # Zero counts
    print(f"\nWith zero counts [0, 1, 3, 2, 0, 3]:")
    iterator = RLEIteratorOptimized([0, 1, 3, 2, 0, 3])
    
    for i in range(5):
        result = iterator.next(1)
        print(f"  next(1): {result}")
    
    # Large skip
    print(f"\nLarge skip [10, 5, 20, 3]:")
    iterator = RLEIteratorQueue([10, 5, 20, 3])
    
    result = iterator.next(15)  # Skip into second run
    print(f"  next(15): {result}")
    
    result = iterator.next(10)  # Skip more
    print(f"  next(10): {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    # Create large encoding
    large_encoding = []
    for i in range(1000):
        large_encoding.extend([100, i % 10])  # 100,000 total elements
    
    implementations = [
        ("Simple Linear", RLEIteratorSimple),
        ("Optimized Binary Search", RLEIteratorOptimized),
        ("Queue-based", RLEIteratorQueue)
    ]
    
    num_operations = 1000
    
    for name, IteratorClass in implementations:
        iterator = IteratorClass(large_encoding)
        
        start_time = time.time()
        
        # Perform many next operations
        for i in range(num_operations):
            n = (i % 10) + 1  # next(1) to next(10)
            result = iterator.next(n)
            
            if result == -1:  # Iterator exhausted
                break
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {name}: {elapsed:.2f}ms for {num_operations} operations")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    encoding = [3, 8, 5, 1, 1, 2, 3, 1]
    iterator = RLEIteratorAdvanced(encoding)
    
    # Test statistics
    initial_stats = iterator.get_statistics()
    print("Initial statistics:")
    for key, value in initial_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test current run info
    print(f"\nCurrent run info:")
    run_info = iterator.get_current_run()
    for key, value in run_info.items():
        print(f"  {key}: {value}")
    
    # Test peek ahead
    print(f"\nPeek ahead 5 elements: {iterator.peek_ahead(5)}")
    
    # Consume some elements
    print(f"\nConsuming elements:")
    for n in [2, 3, 1]:
        result = iterator.next(n)
        print(f"  next({n}): {result}")
        
        # Show updated statistics
        stats = iterator.get_statistics()
        print(f"    Progress: {stats['progress_percentage']:.1f}%")
        
        # Show current run
        run_info = iterator.get_current_run()
        if run_info['run_index'] != -1:
            print(f"    In run {run_info['run_index']}, value: {run_info['run_value']}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Compressed data streaming
    print("Application 1: Compressed Data Streaming")
    
    # Simulate compressed sensor data: [count, reading] pairs
    sensor_data = [100, 25, 50, 26, 200, 25, 75, 24]  # Temperature readings
    
    data_stream = RLEIteratorAdvanced(sensor_data)
    
    print(f"  Compressed sensor data: {sensor_data}")
    print(f"  Total readings: {data_stream.get_statistics()['total_elements']}")
    
    # Stream data in batches
    batch_sizes = [50, 100, 75, 50]
    
    for i, batch_size in enumerate(batch_sizes):
        last_reading = data_stream.next(batch_size)
        stats = data_stream.get_statistics()
        
        print(f"  Batch {i+1} ({batch_size} readings): last value = {last_reading}")
        print(f"    Progress: {stats['progress_percentage']:.1f}%")
    
    # Application 2: Video frame processing
    print(f"\nApplication 2: Video Frame Processing")
    
    # Frames: [duration_in_frames, frame_type_id]
    video_frames = [30, 1, 5, 2, 25, 1, 10, 2, 40, 1]  # 1=normal, 2=action
    frame_processor = RLEIteratorQueue(video_frames)
    
    print(f"  Video encoding: {video_frames}")
    
    # Process frames in real-time (30 fps simulation)
    frame_types = {1: "normal", 2: "action"}
    
    print(f"  Processing frames:")
    for second in range(1, 4):  # Process 3 seconds
        frame_type_id = frame_processor.next(30)  # 30 frames per second
        frame_type = frame_types.get(frame_type_id, "unknown")
        
        print(f"    Second {second}: {frame_type} scenes")
    
    # Application 3: Log data compression
    print(f"\nApplication 3: Log Data Analysis")
    
    # Log levels: [count, level_id] where 1=INFO, 2=WARN, 3=ERROR
    log_data = [1000, 1, 50, 2, 5, 3, 500, 1, 20, 2, 2, 3]
    log_analyzer = RLEIteratorOptimized(log_data)
    
    log_levels = {1: "INFO", 2: "WARN", 3: "ERROR"}
    
    print(f"  Log data encoding: {log_data}")
    
    # Analyze log patterns
    analysis_windows = [100, 200, 300, 400]
    
    for window_size in analysis_windows:
        last_level_id = log_analyzer.next(window_size)
        last_level = log_levels.get(last_level_id, "unknown")
        
        print(f"  Last {window_size} entries end with: {last_level}")

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Create very long but compressible sequence
    long_encoding = []
    
    # Pattern: many of same value, then few of different value
    for i in range(100):
        long_encoding.extend([1000, 1])  # 1000 ones
        long_encoding.extend([10, 2])    # 10 twos
    
    implementations = [
        ("Simple Linear", RLEIteratorSimple),
        ("Optimized", RLEIteratorOptimized),
        ("Queue-based", RLEIteratorQueue),
        ("Advanced", RLEIteratorAdvanced)
    ]
    
    for name, IteratorClass in implementations:
        iterator = IteratorClass(long_encoding)
        
        # Estimate memory usage (simplified)
        if hasattr(iterator, 'runs'):
            memory_estimate = len(iterator.runs) * 4  # Rough estimate
        elif hasattr(iterator, 'queue'):
            memory_estimate = len(iterator.queue) * 2
        elif hasattr(iterator, 'encoding'):
            memory_estimate = len(iterator.encoding)
        else:
            memory_estimate = len(long_encoding)
        
        print(f"  {name}: ~{memory_estimate} memory units")

def test_boundary_conditions():
    """Test boundary conditions"""
    print("\n=== Testing Boundary Conditions ===")
    
    # Very large single run
    print("Very large single run:")
    large_run = [1000000, 42]
    iterator = RLEIteratorOptimized(large_run)
    
    # Skip to near the end
    result = iterator.next(999999)
    print(f"  next(999999): {result}")
    
    result = iterator.next(1)
    print(f"  next(1): {result}")
    
    result = iterator.next(1)
    print(f"  next(1): {result}")
    
    # Many small runs
    print(f"\nMany small runs:")
    small_runs = []
    for i in range(1000):
        small_runs.extend([1, i % 10])
    
    iterator2 = RLEIteratorAdvanced(small_runs)
    
    # Test statistics
    stats = iterator2.get_statistics()
    print(f"  Total elements: {stats['total_elements']}")
    print(f"  Total runs: {stats['total_runs']}")
    
    # Skip through many runs
    result = iterator2.next(500)
    print(f"  next(500): {result}")
    
    updated_stats = iterator2.get_statistics()
    print(f"  Progress: {updated_stats['progress_percentage']:.1f}%")

def stress_test_rle_iterator():
    """Stress test RLE iterator"""
    print("\n=== Stress Testing RLE Iterator ===")
    
    import time
    
    # Create stress test encoding
    stress_encoding = []
    
    # Mix of large and small runs
    for i in range(500):
        if i % 10 == 0:
            stress_encoding.extend([10000, i % 100])  # Large runs
        else:
            stress_encoding.extend([10, i % 100])     # Small runs
    
    print(f"Created encoding with {len(stress_encoding)//2} runs")
    
    iterator = RLEIteratorOptimized(stress_encoding)
    
    start_time = time.time()
    
    # Perform many operations with varying sizes
    operations = 0
    while True:
        n = (operations % 100) + 1  # next(1) to next(100)
        result = iterator.next(n)
        operations += 1
        
        if result == -1 or operations >= 10000:
            break
    
    elapsed = time.time() - start_time
    
    print(f"Performed {operations} operations in {elapsed:.3f}s")
    print(f"Average: {(elapsed/operations)*1000:.3f}ms per operation")

def benchmark_skip_patterns():
    """Benchmark different skip patterns"""
    print("\n=== Benchmarking Skip Patterns ===")
    
    import time
    
    encoding = []
    for i in range(100):
        encoding.extend([100, i % 10])  # 10,000 total elements
    
    patterns = [
        ("Small skips", [1, 2, 3, 4, 5]),
        ("Medium skips", [10, 20, 30, 40, 50]),
        ("Large skips", [100, 200, 300, 400, 500]),
        ("Mixed skips", [1, 50, 5, 200, 10])
    ]
    
    for pattern_name, skip_sizes in patterns:
        iterator = RLEIteratorOptimized(encoding)
        
        start_time = time.time()
        
        # Repeat pattern until exhausted
        cycles = 0
        while True:
            exhausted = False
            for skip_size in skip_sizes:
                result = iterator.next(skip_size)
                if result == -1:
                    exhausted = True
                    break
            
            cycles += 1
            if exhausted or cycles >= 100:
                break
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {pattern_name}: {elapsed:.2f}ms for {cycles} cycles")

if __name__ == "__main__":
    test_rle_iterator_basic()
    test_rle_iterator_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    test_boundary_conditions()
    stress_test_rle_iterator()
    benchmark_skip_patterns()

"""
RLE Iterator Design demonstrates key concepts:

Core Approaches:
1. Simple Linear - Track current run and remaining count
2. Optimized Binary Search - Preprocess cumulative counts
3. Lazy Generator - Use Python generator for element production
4. Queue-based - Process runs using queue data structure  
5. Advanced - Enhanced with statistics and peek functionality

Key Design Principles:
- Run-length encoding compression and decompression
- Efficient skipping through compressed data
- Memory vs time trade-offs in iterator design
- Stateful iteration with position tracking

Performance Characteristics:
- Simple: O(1) space, O(k) time where k is runs to skip
- Optimized: O(n) space, O(log n) time with binary search
- Lazy: O(1) space, O(n) time to consume n elements
- Queue: O(n) space, O(k) time for processing

Real-world Applications:
- Compressed data streaming and processing
- Video frame processing with temporal compression
- Log data analysis with run-length compressed logs
- Sensor data streaming with repeated readings
- Database storage with columnar compression
- Image processing with RLE compressed formats

The optimized binary search approach provides the best
balance for most use cases, offering O(log n) lookup
time with reasonable preprocessing overhead.
"""
