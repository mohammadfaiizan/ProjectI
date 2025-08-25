"""
1656. Design an Ordered Stream - Multiple Approaches
Difficulty: Easy

There is a stream of n (idKey, value) pairs arriving in an arbitrary order, where idKey is an integer between 1 and n and value is a string. No two pairs have the same id.

Design a stream that returns the values in increasing order of their IDs by returning a (possibly empty) list of values after each insertion. The insertion of each pair follows these rules:
- If the stream has no missing ids, insert the pair and return a list of all the consecutive values starting from the smallest missing id.
- If the stream has missing ids, insert the pair and return an empty list.

Implement the OrderedStream class:
- OrderedStream(int n) Constructs the stream to take n values.
- List<String> insert(int idKey, String value) Inserts the pair (idKey, value) into the stream, then returns the largest possible chunk of currently consecutive values starting from the smallest missing id.
"""

from typing import List, Optional, Dict

class OrderedStreamSimple:
    """
    Approach 1: Simple Array-based Implementation
    
    Use array to store values and pointer for next expected ID.
    
    Time Complexity:
    - __init__: O(n)
    - insert: O(k) where k is number of consecutive values returned
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        self.n = n
        self.stream = [None] * (n + 1)  # Index 0 unused, 1-indexed
        self.ptr = 1  # Points to next expected ID
    
    def insert(self, idKey: int, value: str) -> List[str]:
        # Insert the value
        self.stream[idKey] = value
        
        # Check if we can return consecutive values starting from ptr
        result = []
        
        while self.ptr <= self.n and self.stream[self.ptr] is not None:
            result.append(self.stream[self.ptr])
            self.ptr += 1
        
        return result

class OrderedStreamOptimized:
    """
    Approach 2: Optimized with Gap Tracking
    
    Track gaps more efficiently for better performance.
    
    Time Complexity:
    - __init__: O(1)
    - insert: O(k) where k is consecutive values returned
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        self.n = n
        self.values = {}  # Store only inserted values
        self.next_id = 1  # Next expected consecutive ID
    
    def insert(self, idKey: int, value: str) -> List[str]:
        # Store the value
        self.values[idKey] = value
        
        # Check if we can advance next_id
        result = []
        
        while self.next_id in self.values:
            result.append(self.values[self.next_id])
            self.next_id += 1
        
        return result

class OrderedStreamAdvanced:
    """
    Approach 3: Advanced with Analytics and Features
    
    Enhanced stream with statistics and additional functionality.
    
    Time Complexity:
    - __init__: O(1)
    - insert: O(k) where k is consecutive values returned
    
    Space Complexity: O(n + analytics)
    """
    
    def __init__(self, n: int):
        self.n = n
        self.values = {}
        self.next_id = 1
        
        # Analytics
        self.total_insertions = 0
        self.total_values_returned = 0
        self.insertion_order = []  # Track insertion order
        self.chunks_returned = []  # Track chunks returned
        
        # Additional features
        self.completed = False
        self.completion_time = None
    
    def insert(self, idKey: int, value: str) -> List[str]:
        import time
        
        self.total_insertions += 1
        self.insertion_order.append((idKey, value, time.time()))
        
        # Store the value
        self.values[idKey] = value
        
        # Find consecutive values starting from next_id
        result = []
        
        while self.next_id in self.values:
            result.append(self.values[self.next_id])
            self.next_id += 1
        
        # Update statistics
        if result:
            self.total_values_returned += len(result)
            self.chunks_returned.append((len(result), list(result)))
        
        # Check for completion
        if self.next_id > self.n and not self.completed:
            self.completed = True
            self.completion_time = time.time()
        
        return result
    
    def get_statistics(self) -> dict:
        """Get stream statistics"""
        progress = (self.next_id - 1) / self.n if self.n > 0 else 0
        
        missing_ids = []
        for i in range(1, self.n + 1):
            if i not in self.values:
                missing_ids.append(i)
        
        return {
            'total_insertions': self.total_insertions,
            'total_values_returned': self.total_values_returned,
            'progress': progress,
            'completed': self.completed,
            'missing_count': len(missing_ids),
            'missing_ids': missing_ids[:10],  # Show first 10 missing
            'chunks_returned_count': len(self.chunks_returned),
            'average_chunk_size': self.total_values_returned / max(1, len(self.chunks_returned))
        }
    
    def get_missing_ids(self) -> List[int]:
        """Get all missing IDs"""
        missing = []
        for i in range(1, self.n + 1):
            if i not in self.values:
                missing.append(i)
        return missing
    
    def get_insertion_history(self) -> List[tuple]:
        """Get insertion history with timestamps"""
        return self.insertion_order.copy()
    
    def peek_next_chunk_size(self, idKey: int, value: str) -> int:
        """Peek at how many values would be returned if this pair was inserted"""
        # Temporarily insert and check
        temp_values = self.values.copy()
        temp_values[idKey] = value
        
        temp_next_id = self.next_id
        count = 0
        
        while temp_next_id in temp_values:
            count += 1
            temp_next_id += 1
        
        return count

class OrderedStreamBuffered:
    """
    Approach 4: Buffered Stream with Batch Processing
    
    Buffer operations for batch processing efficiency.
    
    Time Complexity:
    - __init__: O(1)
    - insert: O(1) amortized
    
    Space Complexity: O(n + buffer_size)
    """
    
    def __init__(self, n: int, buffer_size: int = 100):
        self.n = n
        self.values = {}
        self.next_id = 1
        
        # Buffering
        self.buffer = []
        self.buffer_size = buffer_size
        self.pending_results = []
    
    def insert(self, idKey: int, value: str) -> List[str]:
        # Add to buffer
        self.buffer.append((idKey, value))
        
        # Process buffer if full or if this might trigger a return
        if len(self.buffer) >= self.buffer_size or idKey == self.next_id:
            return self._process_buffer()
        
        return []
    
    def _process_buffer(self) -> List[str]:
        """Process buffered insertions"""
        # Insert all buffered values
        for idKey, value in self.buffer:
            self.values[idKey] = value
        
        self.buffer = []
        
        # Find consecutive values
        result = []
        
        while self.next_id in self.values:
            result.append(self.values[self.next_id])
            self.next_id += 1
        
        return result
    
    def flush(self) -> List[str]:
        """Force process remaining buffer"""
        if self.buffer:
            return self._process_buffer()
        return []

class OrderedStreamConcurrent:
    """
    Approach 5: Thread-Safe Ordered Stream
    
    Concurrent implementation for multi-threaded access.
    
    Time Complexity:
    - __init__: O(1)
    - insert: O(k) + lock overhead
    
    Space Complexity: O(n)
    """
    
    def __init__(self, n: int):
        import threading
        
        self.n = n
        self.values = {}
        self.next_id = 1
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.concurrent_insertions = 0
        self.total_wait_time = 0
    
    def insert(self, idKey: int, value: str) -> List[str]:
        import time
        
        wait_start = time.time()
        
        with self.lock:
            wait_time = time.time() - wait_start
            self.total_wait_time += wait_time
            self.concurrent_insertions += 1
            
            # Store value
            self.values[idKey] = value
            
            # Find consecutive values
            result = []
            
            while self.next_id in self.values:
                result.append(self.values[self.next_id])
                self.next_id += 1
            
            return result
    
    def get_concurrency_stats(self) -> dict:
        """Get concurrency statistics"""
        with self.lock:
            avg_wait_time = self.total_wait_time / max(1, self.concurrent_insertions)
            
            return {
                'concurrent_insertions': self.concurrent_insertions,
                'total_wait_time': self.total_wait_time,
                'average_wait_time': avg_wait_time,
                'current_progress': self.next_id - 1,
                'remaining': self.n - (self.next_id - 1)
            }


def test_ordered_stream_basic():
    """Test basic ordered stream functionality"""
    print("=== Testing Basic Ordered Stream Functionality ===")
    
    implementations = [
        ("Simple", OrderedStreamSimple),
        ("Optimized", OrderedStreamOptimized),
        ("Advanced", OrderedStreamAdvanced),
        ("Buffered", lambda n: OrderedStreamBuffered(n, 2)),
        ("Concurrent", OrderedStreamConcurrent)
    ]
    
    for name, StreamClass in implementations:
        print(f"\n{name}:")
        
        stream = StreamClass(5)
        
        # Test sequence from problem
        insertions = [
            (3, "ccc"),
            (1, "aaa"), 
            (2, "bbb"),
            (5, "eee"),
            (4, "ddd")
        ]
        
        for idKey, value in insertions:
            result = stream.insert(idKey, value)
            print(f"  insert({idKey}, '{value}'): {result}")

def test_ordered_stream_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Ordered Stream Edge Cases ===")
    
    # Test single element stream
    print("Single element stream:")
    stream = OrderedStreamSimple(1)
    
    result = stream.insert(1, "only")
    print(f"  insert(1, 'only'): {result}")
    
    # Test reverse order insertion
    print(f"\nReverse order insertion:")
    stream = OrderedStreamOptimized(4)
    
    reverse_insertions = [(4, "d"), (3, "c"), (2, "b"), (1, "a")]
    
    for idKey, value in reverse_insertions:
        result = stream.insert(idKey, value)
        print(f"  insert({idKey}, '{value}'): {result}")
    
    # Test scattered insertions
    print(f"\nScattered insertions:")
    stream = OrderedStreamAdvanced(6)
    
    scattered = [(6, "f"), (1, "a"), (4, "d"), (2, "b"), (5, "e"), (3, "c")]
    
    for idKey, value in scattered:
        result = stream.insert(idKey, value)
        print(f"  insert({idKey}, '{value}'): {result}")
    
    # Test duplicate insertions (if implementation allows)
    print(f"\nTesting completion status:")
    if hasattr(stream, 'get_statistics'):
        final_stats = stream.get_statistics()
        print(f"  Completed: {final_stats['completed']}")
        print(f"  Progress: {final_stats['progress']:.1%}")

def test_performance_patterns():
    """Test different insertion patterns"""
    print("\n=== Testing Performance Patterns ===")
    
    import time
    
    patterns = [
        ("Sequential", lambda n: [(i, f"val{i}") for i in range(1, n+1)]),
        ("Reverse", lambda n: [(i, f"val{i}") for i in range(n, 0, -1)]),
        ("Random", lambda n: [(i, f"val{i}") for i in [3, 1, 5, 2, 4]]),
        ("Gaps First", lambda n: [(i, f"val{i}") for i in [2, 4, 1, 3, 5]])
    ]
    
    stream_size = 5
    
    for pattern_name, pattern_func in patterns:
        print(f"\n{pattern_name} pattern:")
        
        insertions = pattern_func(stream_size)
        
        implementations = [
            ("Simple", OrderedStreamSimple),
            ("Optimized", OrderedStreamOptimized)
        ]
        
        for impl_name, StreamClass in implementations:
            stream = StreamClass(stream_size)
            
            start_time = time.time()
            
            total_returned = 0
            chunks_count = 0
            
            for idKey, value in insertions:
                result = stream.insert(idKey, value)
                total_returned += len(result)
                if result:
                    chunks_count += 1
            
            elapsed = (time.time() - start_time) * 1000
            
            print(f"  {impl_name}: {elapsed:.3f}ms, {total_returned} values, {chunks_count} chunks")

def test_advanced_features():
    """Test advanced stream features"""
    print("\n=== Testing Advanced Features ===")
    
    stream = OrderedStreamAdvanced(8)
    
    # Insert some values with gaps
    insertions = [
        (2, "second"), (5, "fifth"), (1, "first"),
        (4, "fourth"), (7, "seventh"), (3, "third")
    ]
    
    print("Building stream with gaps:")
    for idKey, value in insertions:
        result = stream.insert(idKey, value)
        print(f"  insert({idKey}, '{value}'): {result}")
    
    # Test statistics
    stats = stream.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        elif isinstance(value, list) and len(value) > 10:
            print(f"  {key}: {value[:5]}... (showing first 5)")
        else:
            print(f"  {key}: {value}")
    
    # Test missing IDs
    missing = stream.get_missing_ids()
    print(f"\nMissing IDs: {missing}")
    
    # Test peek functionality
    print(f"\nPeek functionality:")
    for test_id in missing[:3]:  # Test first few missing
        peek_size = stream.peek_next_chunk_size(test_id, f"test{test_id}")
        print(f"  If insert({test_id}, 'test{test_id}'): would return {peek_size} values")
    
    # Complete the stream
    print(f"\nCompleting the stream:")
    for missing_id in missing:
        result = stream.insert(missing_id, f"missing{missing_id}")
        if result:
            print(f"  insert({missing_id}, 'missing{missing_id}'): {len(result)} values")
    
    # Final statistics
    final_stats = stream.get_statistics()
    print(f"\nFinal status:")
    print(f"  Completed: {final_stats['completed']}")
    print(f"  Total chunks: {final_stats['chunks_returned_count']}")
    print(f"  Average chunk size: {final_stats['average_chunk_size']:.1f}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Log processing system
    print("Application 1: Distributed Log Processing")
    
    log_stream = OrderedStreamAdvanced(10)
    
    # Simulate log entries arriving out of order
    log_entries = [
        (3, "User login attempt"),
        (1, "System startup"),
        (5, "Database connection established"),
        (2, "Configuration loaded"),
        (4, "Service initialized"),
        (7, "First client request"),
        (6, "Cache warmed up"),
        (9, "Peak traffic detected"),
        (8, "Load balancer activated"),
        (10, "System stabilized")
    ]
    
    print("  Processing log entries (out of order):")
    
    for seq_id, log_message in log_entries:
        processed_logs = log_stream.insert(seq_id, log_message)
        
        print(f"    Received log {seq_id}: '{log_message}'")
        
        if processed_logs:
            print(f"      Processed batch: {len(processed_logs)} logs")
            for i, log in enumerate(processed_logs):
                print(f"        {i+1}. {log}")
    
    # Application 2: Video frame processing
    print(f"\nApplication 2: Video Frame Reconstruction")
    
    frame_stream = OrderedStreamBuffered(8, 3)  # 8 frames, buffer size 3
    
    # Simulate video frames arriving out of order
    frames = [
        (4, "Frame_004.jpg"),
        (1, "Frame_001.jpg"),
        (6, "Frame_006.jpg"),
        (2, "Frame_002.jpg"),
        (8, "Frame_008.jpg"),
        (3, "Frame_003.jpg"),
        (5, "Frame_005.jpg"),
        (7, "Frame_007.jpg")
    ]
    
    print("  Processing video frames:")
    
    for frame_id, frame_file in frames:
        ready_frames = frame_stream.insert(frame_id, frame_file)
        
        print(f"    Received frame {frame_id}: {frame_file}")
        
        if ready_frames:
            print(f"      Ready for playback: {ready_frames}")
    
    # Flush any remaining buffered frames
    remaining_frames = frame_stream.flush()
    if remaining_frames:
        print(f"    Final flush: {remaining_frames}")
    
    # Application 3: Message ordering in distributed system
    print(f"\nApplication 3: Distributed Message Ordering")
    
    message_stream = OrderedStreamConcurrent(6)
    
    # Simulate messages from different nodes
    messages = [
        (3, "Node_B: Data update"),
        (1, "Node_A: Transaction start"),
        (5, "Node_C: Validation complete"),
        (2, "Node_A: Lock acquired"),
        (4, "Node_B: Processing done"),
        (6, "Node_C: Transaction commit")
    ]
    
    print("  Processing distributed messages:")
    
    for msg_id, message_content in messages:
        ordered_messages = message_stream.insert(msg_id, message_content)
        
        print(f"    Message {msg_id}: {message_content}")
        
        if ordered_messages:
            print(f"      Ordered sequence ready:")
            for i, msg in enumerate(ordered_messages):
                print(f"        Step {i+1}: {msg}")
    
    # Application 4: Package delivery tracking
    print(f"\nApplication 4: Package Delivery Sequence")
    
    delivery_stream = OrderedStreamOptimized(5)
    
    # Simulate delivery checkpoints
    checkpoints = [
        (2, "Warehouse_departure"),
        (4, "Local_facility_arrival"),
        (1, "Order_confirmed"),
        (5, "Delivered_to_customer"),
        (3, "In_transit_to_destination")
    ]
    
    print("  Tracking delivery progress:")
    
    for checkpoint_id, status in checkpoints:
        completed_steps = delivery_stream.insert(checkpoint_id, status)
        
        print(f"    Checkpoint {checkpoint_id}: {status}")
        
        if completed_steps:
            print(f"      Delivery progress update:")
            for i, step in enumerate(completed_steps):
                print(f"        {i+1}. {step}")

def test_concurrent_access():
    """Test concurrent access patterns"""
    print("\n=== Testing Concurrent Access ===")
    
    import threading
    import time
    import random
    
    stream = OrderedStreamConcurrent(20)
    
    # Test concurrent insertions
    num_threads = 4
    insertions_per_thread = 5
    
    def insert_worker(thread_id: int, results: list):
        """Worker thread for concurrent insertions"""
        thread_results = []
        
        # Each thread inserts a subset of IDs
        start_id = thread_id * insertions_per_thread + 1
        end_id = start_id + insertions_per_thread
        
        for i in range(start_id, end_id):
            # Add some randomness to insertion timing
            time.sleep(random.uniform(0.001, 0.005))
            
            result = stream.insert(i, f"value_{i}_from_thread_{thread_id}")
            thread_results.append((i, len(result)))
        
        results.append(thread_results)
    
    print(f"Testing {num_threads} concurrent threads...")
    
    # Start threads
    threads = []
    results = []
    
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(target=insert_worker, args=(i, results))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    elapsed_time = time.time() - start_time
    
    # Analyze results
    total_insertions = num_threads * insertions_per_thread
    total_chunks = sum(sum(len(result) for _, result in thread_results) for thread_results in results)
    
    print(f"  Completed {total_insertions} insertions in {elapsed_time:.3f}s")
    print(f"  Total chunks returned: {total_chunks}")
    
    # Get concurrency statistics
    concurrency_stats = stream.get_concurrency_stats()
    print(f"  Concurrency stats:")
    for key, value in concurrency_stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

def stress_test_ordered_stream():
    """Stress test ordered stream"""
    print("\n=== Stress Testing Ordered Stream ===")
    
    import time
    import random
    
    # Large stream test
    stream_size = 10000
    stream = OrderedStreamOptimized(stream_size)
    
    print(f"Stress test: {stream_size} elements")
    
    # Generate random insertion order
    insertion_order = list(range(1, stream_size + 1))
    random.shuffle(insertion_order)
    
    start_time = time.time()
    
    total_values_returned = 0
    chunks_returned = 0
    
    # Insert all elements
    for i, idKey in enumerate(insertion_order):
        result = stream.insert(idKey, f"value_{idKey}")
        
        total_values_returned += len(result)
        if result:
            chunks_returned += 1
        
        # Progress update every 1000 insertions
        if (i + 1) % 1000 == 0:
            progress = (i + 1) / stream_size
            print(f"    Progress: {progress:.1%}")
    
    elapsed_time = time.time() - start_time
    
    print(f"  Completed in {elapsed_time:.2f}s")
    print(f"  Rate: {stream_size / elapsed_time:.0f} insertions/sec")
    print(f"  Total values returned: {total_values_returned}")
    print(f"  Chunks returned: {chunks_returned}")
    print(f"  Average chunk size: {total_values_returned / max(1, chunks_returned):.1f}")
    
    # Verify all values were returned
    if total_values_returned == stream_size:
        print("  ✓ All values successfully returned")
    else:
        print(f"  ✗ Missing values: {stream_size - total_values_returned}")

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Simple", OrderedStreamSimple),
        ("Optimized", OrderedStreamOptimized),
        ("Advanced", OrderedStreamAdvanced)
    ]
    
    stream_size = 5000
    
    for name, StreamClass in implementations:
        stream = StreamClass(stream_size)
        
        # Insert half the elements randomly
        import random
        random.seed(42)  # Consistent results
        
        insert_count = stream_size // 2
        ids_to_insert = random.sample(range(1, stream_size + 1), insert_count)
        
        for idKey in ids_to_insert:
            stream.insert(idKey, f"value_{idKey}")
        
        # Estimate memory usage
        if hasattr(stream, 'stream') and stream.stream:
            # Simple approach with array
            memory_estimate = len(stream.stream)
        elif hasattr(stream, 'values'):
            # Dictionary-based approaches
            memory_estimate = len(stream.values)
        else:
            memory_estimate = insert_count
        
        print(f"  {name}:")
        print(f"    Inserted: {insert_count} / {stream_size}")
        print(f"    Memory estimate: {memory_estimate} units")
        
        if hasattr(stream, 'get_statistics'):
            stats = stream.get_statistics()
            print(f"    Missing IDs: {stats['missing_count']}")

def benchmark_chunk_sizes():
    """Benchmark different chunk size patterns"""
    print("\n=== Benchmarking Chunk Sizes ===")
    
    import time
    
    # Different insertion patterns that create different chunk sizes
    patterns = [
        ("Small gaps", [1, 3, 2, 5, 4, 7, 6, 8]),  # Alternating pattern
        ("Large chunks", [5, 6, 7, 8, 1, 2, 3, 4]),  # Two large chunks
        ("Single elements", [2, 4, 6, 8, 1, 3, 5, 7]),  # Individual elements
        ("Sequential", [1, 2, 3, 4, 5, 6, 7, 8])  # Perfect order
    ]
    
    for pattern_name, insertion_order in patterns:
        stream = OrderedStreamAdvanced(8)
        
        start_time = time.time()
        
        chunk_sizes = []
        
        for idKey in insertion_order:
            result = stream.insert(idKey, f"val{idKey}")
            if result:
                chunk_sizes.append(len(result))
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"  {pattern_name}:")
        print(f"    Time: {elapsed:.3f}ms")
        print(f"    Chunk sizes: {chunk_sizes}")
        print(f"    Total chunks: {len(chunk_sizes)}")
        print(f"    Average chunk size: {sum(chunk_sizes) / max(1, len(chunk_sizes)):.1f}")

if __name__ == "__main__":
    test_ordered_stream_basic()
    test_ordered_stream_edge_cases()
    test_performance_patterns()
    test_advanced_features()
    demonstrate_applications()
    test_concurrent_access()
    stress_test_ordered_stream()
    test_memory_efficiency()
    benchmark_chunk_sizes()

"""
Ordered Stream Design demonstrates key concepts:

Core Approaches:
1. Simple - Array-based storage with pointer tracking
2. Optimized - Dictionary storage for sparse data
3. Advanced - Enhanced with analytics and additional features
4. Buffered - Batch processing for improved efficiency
5. Concurrent - Thread-safe implementation for multi-user access

Key Design Principles:
- Stream processing with ordering constraints
- Gap detection and consecutive sequence identification
- Memory vs time trade-offs for different use cases
- Real-time processing vs batch processing strategies

Performance Characteristics:
- Simple: O(n) space always, O(k) time per insert
- Optimized: O(m) space where m is inserted elements, O(k) time
- Advanced: Additional overhead for analytics and features
- Buffered: Amortized performance with batch processing

Real-world Applications:
- Distributed log processing and ordering
- Video frame reconstruction from network packets
- Message ordering in distributed systems
- Package delivery tracking and sequencing
- Database transaction log ordering
- Real-time data stream processing

The optimized approach provides the best balance
for most use cases, offering efficient memory usage
while maintaining good performance for gap detection
and consecutive sequence processing.
"""
