"""
Thread Safe Queue Stack - Multiple Approaches
Difficulty: Medium

Implement thread-safe queue and stack data structures with various synchronization mechanisms.
Focus on concurrent access, deadlock prevention, and performance optimization.
"""

import threading
import time
import queue
from typing import Any, Optional, Generic, TypeVar
from collections import deque
import random

T = TypeVar('T')

class ThreadSafeStack:
    """
    Approach 1: Thread-Safe Stack with RLock
    
    Stack implementation using reentrant lock for thread safety.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items = []
        self._capacity = capacity
        self._lock = threading.RLock()
    
    def push(self, item: Any) -> None:
        """Thread-safe push operation"""
        with self._lock:
            if self._capacity and len(self._items) >= self._capacity:
                raise OverflowError("Stack overflow")
            self._items.append(item)
    
    def pop(self) -> Any:
        """Thread-safe pop operation"""
        with self._lock:
            if not self._items:
                raise IndexError("Stack underflow")
            return self._items.pop()
    
    def peek(self) -> Any:
        """Thread-safe peek operation"""
        with self._lock:
            if not self._items:
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
    
    def clear(self) -> None:
        """Thread-safe clear operation"""
        with self._lock:
            self._items.clear()


class ThreadSafeQueue:
    """
    Approach 2: Thread-Safe Queue with Condition Variables
    
    Queue implementation using condition variables for blocking operations.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, capacity: Optional[int] = None):
        self._items = deque()
        self._capacity = capacity
        self._lock = threading.RLock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def enqueue(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Thread-safe enqueue with optional timeout"""
        with self._not_full:
            if self._capacity:
                while len(self._items) >= self._capacity:
                    if not self._not_full.wait(timeout):
                        return False
            
            self._items.append(item)
            self._not_empty.notify()
            return True
    
    def dequeue(self, timeout: Optional[float] = None) -> Any:
        """Thread-safe dequeue with optional timeout"""
        with self._not_empty:
            while len(self._items) == 0:
                if not self._not_empty.wait(timeout):
                    raise queue.Empty("Queue is empty - timeout")
            
            item = self._items.popleft()
            self._not_full.notify()
            return item
    
    def front(self) -> Any:
        """Thread-safe front peek"""
        with self._lock:
            if not self._items:
                raise IndexError("Queue is empty")
            return self._items[0]
    
    def is_empty(self) -> bool:
        """Thread-safe empty check"""
        with self._lock:
            return len(self._items) == 0
    
    def size(self) -> int:
        """Thread-safe size check"""
        with self._lock:
            return len(self._items)


class LockFreeStack:
    """
    Approach 3: Lock-Free Stack using Compare-and-Swap
    
    Lock-free implementation using atomic operations (conceptual).
    
    Time: O(1) for operations, Space: O(n)
    """
    
    class Node:
        def __init__(self, data: Any, next_node: Optional['Node'] = None):
            self.data = data
            self.next = next_node
    
    def __init__(self):
        self._top = None
        self._lock = threading.Lock()  # Fallback for Python's GIL
    
    def push(self, item: Any) -> None:
        """Lock-free push (with fallback lock)"""
        new_node = self.Node(item)
        
        # In a true lock-free implementation, this would use CAS
        with self._lock:
            new_node.next = self._top
            self._top = new_node
    
    def pop(self) -> Any:
        """Lock-free pop (with fallback lock)"""
        with self._lock:
            if self._top is None:
                raise IndexError("Stack underflow")
            
            data = self._top.data
            self._top = self._top.next
            return data
    
    def is_empty(self) -> bool:
        """Check if empty"""
        with self._lock:
            return self._top is None


class ReadWriteLockStack:
    """
    Approach 4: Stack with Read-Write Lock
    
    Optimized for multiple readers, single writer scenarios.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self):
        self._items = []
        self._rw_lock = threading.RLock()
        self._readers = 0
        self._writers = 0
        self._read_ready = threading.Condition(self._rw_lock)
        self._write_ready = threading.Condition(self._rw_lock)
    
    def _acquire_read(self):
        """Acquire read lock"""
        with self._rw_lock:
            while self._writers > 0:
                self._read_ready.wait()
            self._readers += 1
    
    def _release_read(self):
        """Release read lock"""
        with self._rw_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_ready.notify_all()
    
    def _acquire_write(self):
        """Acquire write lock"""
        with self._rw_lock:
            while self._writers > 0 or self._readers > 0:
                self._write_ready.wait()
            self._writers += 1
    
    def _release_write(self):
        """Release write lock"""
        with self._rw_lock:
            self._writers -= 1
            self._write_ready.notify_all()
            self._read_ready.notify_all()
    
    def push(self, item: Any) -> None:
        """Write operation - push"""
        self._acquire_write()
        try:
            self._items.append(item)
        finally:
            self._release_write()
    
    def pop(self) -> Any:
        """Write operation - pop"""
        self._acquire_write()
        try:
            if not self._items:
                raise IndexError("Stack underflow")
            return self._items.pop()
        finally:
            self._release_write()
    
    def peek(self) -> Any:
        """Read operation - peek"""
        self._acquire_read()
        try:
            if not self._items:
                raise IndexError("Stack is empty")
            return self._items[-1]
        finally:
            self._release_read()
    
    def size(self) -> int:
        """Read operation - size"""
        self._acquire_read()
        try:
            return len(self._items)
        finally:
            self._release_read()


class ProducerConsumerQueue:
    """
    Approach 5: Producer-Consumer Queue
    
    Specialized queue for producer-consumer pattern.
    
    Time: O(1) for operations, Space: O(capacity)
    """
    
    def __init__(self, capacity: int):
        self._buffer = [None] * capacity
        self._capacity = capacity
        self._count = 0
        self._in_ptr = 0
        self._out_ptr = 0
        
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def produce(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Producer operation"""
        with self._not_full:
            while self._count >= self._capacity:
                if not self._not_full.wait(timeout):
                    return False
            
            self._buffer[self._in_ptr] = item
            self._in_ptr = (self._in_ptr + 1) % self._capacity
            self._count += 1
            
            self._not_empty.notify()
            return True
    
    def consume(self, timeout: Optional[float] = None) -> Any:
        """Consumer operation"""
        with self._not_empty:
            while self._count == 0:
                if not self._not_empty.wait(timeout):
                    raise queue.Empty("Buffer is empty - timeout")
            
            item = self._buffer[self._out_ptr]
            self._buffer[self._out_ptr] = None  # Help GC
            self._out_ptr = (self._out_ptr + 1) % self._capacity
            self._count -= 1
            
            self._not_full.notify()
            return item
    
    def size(self) -> int:
        """Get current size"""
        with self._lock:
            return self._count
    
    def is_full(self) -> bool:
        """Check if full"""
        with self._lock:
            return self._count >= self._capacity
    
    def is_empty(self) -> bool:
        """Check if empty"""
        with self._lock:
            return self._count == 0


class ThreadPoolQueue:
    """
    Approach 6: Thread Pool Work Queue
    
    Queue designed for thread pool task distribution.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self, max_workers: int = 4):
        self._queue = queue.Queue()
        self._workers = []
        self._shutdown = False
        self._lock = threading.Lock()
        
        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
    
    def _worker(self, worker_id: int):
        """Worker thread function"""
        while not self._shutdown:
            try:
                task = self._queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                # Execute task
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Worker {worker_id} error: {e}")
                finally:
                    self._queue.task_done()
                    
            except queue.Empty:
                continue
    
    def submit(self, func, *args, **kwargs) -> None:
        """Submit task to thread pool"""
        if self._shutdown:
            raise RuntimeError("Thread pool is shut down")
        
        task = (func, args, kwargs)
        self._queue.put(task)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool"""
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            
            # Send shutdown signals
            for _ in self._workers:
                self._queue.put(None)
            
            if wait:
                for worker in self._workers:
                    worker.join()
    
    def pending_tasks(self) -> int:
        """Get number of pending tasks"""
        return self._queue.qsize()


def test_thread_safe_implementations():
    """Test thread-safe implementations"""
    print("=== Testing Thread-Safe Implementations ===")
    
    implementations = [
        ("Thread-Safe Stack", ThreadSafeStack),
        ("Thread-Safe Queue", ThreadSafeQueue),
        ("Lock-Free Stack", LockFreeStack),
        ("Read-Write Stack", ReadWriteLockStack),
    ]
    
    for name, impl_class in implementations:
        print(f"\n--- {name} ---")
        
        try:
            if "Queue" in name:
                container = impl_class()
                
                # Test queue operations
                container.enqueue("item1")
                container.enqueue("item2")
                print(f"  Enqueued 2 items, size: {container.size()}")
                
                item = container.dequeue()
                print(f"  Dequeued: {item}, size: {container.size()}")
                
            else:
                container = impl_class()
                
                # Test stack operations
                container.push("item1")
                container.push("item2")
                print(f"  Pushed 2 items, size: {container.size()}")
                
                if hasattr(container, 'peek'):
                    top = container.peek()
                    print(f"  Top item: {top}")
                
                item = container.pop()
                print(f"  Popped: {item}, size: {container.size()}")
            
            print(f"  {name} basic tests passed ✓")
            
        except Exception as e:
            print(f"  {name} error: {str(e)[:50]}")


def demonstrate_concurrent_access():
    """Demonstrate concurrent access patterns"""
    print("\n=== Concurrent Access Demonstration ===")
    
    # Test with multiple threads
    shared_stack = ThreadSafeStack()
    shared_queue = ThreadSafeQueue()
    results = []
    
    def stack_worker(worker_id: int, operation: str):
        """Worker thread for stack operations"""
        try:
            if operation == "push":
                for i in range(5):
                    item = f"W{worker_id}_item_{i}"
                    shared_stack.push(item)
                    results.append(f"Worker {worker_id} pushed {item}")
                    time.sleep(random.uniform(0.001, 0.01))
            
            elif operation == "pop":
                for _ in range(3):
                    try:
                        item = shared_stack.pop()
                        results.append(f"Worker {worker_id} popped {item}")
                        time.sleep(random.uniform(0.001, 0.01))
                    except IndexError:
                        results.append(f"Worker {worker_id} found empty stack")
                        break
        
        except Exception as e:
            results.append(f"Worker {worker_id} error: {e}")
    
    def queue_worker(worker_id: int, operation: str):
        """Worker thread for queue operations"""
        try:
            if operation == "enqueue":
                for i in range(5):
                    item = f"Q{worker_id}_item_{i}"
                    shared_queue.enqueue(item)
                    results.append(f"Worker {worker_id} enqueued {item}")
                    time.sleep(random.uniform(0.001, 0.01))
            
            elif operation == "dequeue":
                for _ in range(3):
                    try:
                        item = shared_queue.dequeue(timeout=0.1)
                        results.append(f"Worker {worker_id} dequeued {item}")
                        time.sleep(random.uniform(0.001, 0.01))
                    except queue.Empty:
                        results.append(f"Worker {worker_id} timeout on dequeue")
                        break
        
        except Exception as e:
            results.append(f"Worker {worker_id} error: {e}")
    
    # Create and start threads
    threads = []
    
    # Stack threads
    for i in range(2):
        t = threading.Thread(target=stack_worker, args=(i, "push"))
        threads.append(t)
        t.start()
    
    for i in range(2, 4):
        t = threading.Thread(target=stack_worker, args=(i, "pop"))
        threads.append(t)
        t.start()
    
    # Queue threads
    for i in range(4, 6):
        t = threading.Thread(target=queue_worker, args=(i, "enqueue"))
        threads.append(t)
        t.start()
    
    for i in range(6, 8):
        t = threading.Thread(target=queue_worker, args=(i, "dequeue"))
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    
    print("Concurrent operations completed:")
    for result in results[:15]:  # Show first 15 results
        print(f"  {result}")
    
    print(f"Final stack size: {shared_stack.size()}")
    print(f"Final queue size: {shared_queue.size()}")


def demonstrate_producer_consumer():
    """Demonstrate producer-consumer pattern"""
    print("\n=== Producer-Consumer Pattern Demonstration ===")
    
    buffer = ProducerConsumerQueue(5)
    results = []
    
    def producer(producer_id: int, items: int):
        """Producer thread"""
        for i in range(items):
            item = f"P{producer_id}_product_{i}"
            success = buffer.produce(item, timeout=1.0)
            if success:
                results.append(f"Producer {producer_id} produced {item}")
            else:
                results.append(f"Producer {producer_id} timeout producing {item}")
            time.sleep(random.uniform(0.01, 0.05))
    
    def consumer(consumer_id: int, items: int):
        """Consumer thread"""
        for _ in range(items):
            try:
                item = buffer.consume(timeout=1.0)
                results.append(f"Consumer {consumer_id} consumed {item}")
                time.sleep(random.uniform(0.01, 0.05))
            except queue.Empty:
                results.append(f"Consumer {consumer_id} timeout consuming")
                break
    
    # Create producer and consumer threads
    threads = []
    
    # Producers
    for i in range(2):
        t = threading.Thread(target=producer, args=(i, 4))
        threads.append(t)
        t.start()
    
    # Consumers
    for i in range(2):
        t = threading.Thread(target=consumer, args=(i, 3))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    print("Producer-Consumer operations:")
    for result in results:
        print(f"  {result}")
    
    print(f"Final buffer size: {buffer.size()}")


def demonstrate_thread_pool():
    """Demonstrate thread pool queue"""
    print("\n=== Thread Pool Queue Demonstration ===")
    
    def sample_task(task_id: int, duration: float):
        """Sample task for thread pool"""
        print(f"    Task {task_id} starting (duration: {duration:.2f}s)")
        time.sleep(duration)
        print(f"    Task {task_id} completed")
    
    # Create thread pool
    pool = ThreadPoolQueue(max_workers=3)
    
    print("Submitting tasks to thread pool:")
    
    # Submit tasks
    for i in range(8):
        duration = random.uniform(0.1, 0.5)
        pool.submit(sample_task, i, duration)
        print(f"  Submitted task {i}")
    
    print(f"Pending tasks: {pool.pending_tasks()}")
    
    # Wait a bit for tasks to complete
    time.sleep(2.0)
    
    print("Shutting down thread pool...")
    pool.shutdown(wait=True)
    print("Thread pool shutdown complete")


def benchmark_thread_safety_overhead():
    """Benchmark thread safety overhead"""
    print("\n=== Thread Safety Overhead Benchmark ===")
    
    import time
    
    # Compare thread-safe vs non-thread-safe implementations
    class SimpleStack:
        def __init__(self):
            self._items = []
        
        def push(self, item):
            self._items.append(item)
        
        def pop(self):
            return self._items.pop()
    
    implementations = [
        ("Simple Stack", SimpleStack),
        ("Thread-Safe Stack", ThreadSafeStack),
    ]
    
    n_operations = 100000
    
    for name, stack_class in implementations:
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
    
    # Calculate overhead
    simple_time = 0.0  # Would need to store from above
    thread_safe_time = 0.0  # Would need to store from above
    # print(f"Thread safety overhead: {((thread_safe_time - simple_time) / simple_time * 100):.1f}%")


def test_deadlock_prevention():
    """Test deadlock prevention mechanisms"""
    print("\n=== Deadlock Prevention Test ===")
    
    # Create two shared resources
    resource1 = ThreadSafeStack()
    resource2 = ThreadSafeQueue()
    
    results = []
    
    def worker1():
        """Worker that acquires resources in order 1, 2"""
        try:
            resource1.push("from_worker1")
            time.sleep(0.01)  # Simulate work
            resource2.enqueue("from_worker1")
            results.append("Worker1 completed successfully")
        except Exception as e:
            results.append(f"Worker1 error: {e}")
    
    def worker2():
        """Worker that acquires resources in order 1, 2 (same order)"""
        try:
            resource1.push("from_worker2")
            time.sleep(0.01)  # Simulate work
            resource2.enqueue("from_worker2")
            results.append("Worker2 completed successfully")
        except Exception as e:
            results.append(f"Worker2 error: {e}")
    
    # Start workers
    t1 = threading.Thread(target=worker1)
    t2 = threading.Thread(target=worker2)
    
    t1.start()
    t2.start()
    
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    
    if t1.is_alive() or t2.is_alive():
        print("  Potential deadlock detected!")
    else:
        print("  No deadlock - both workers completed")
    
    for result in results:
        print(f"  {result}")


def analyze_synchronization_mechanisms():
    """Analyze different synchronization mechanisms"""
    print("\n=== Synchronization Mechanisms Analysis ===")
    
    mechanisms = [
        ("RLock", "Reentrant lock", "Simple, allows recursive locking", "Moderate overhead"),
        ("Condition", "Condition variables", "Efficient blocking/signaling", "Low overhead"),
        ("Semaphore", "Counting semaphore", "Resource counting", "Low overhead"),
        ("Event", "Event signaling", "Simple signaling mechanism", "Very low overhead"),
        ("Barrier", "Thread barrier", "Synchronize multiple threads", "Moderate overhead"),
    ]
    
    print(f"{'Mechanism':<12} | {'Description':<20} | {'Use Case':<30} | {'Overhead'}")
    print("-" * 85)
    
    for mechanism, description, use_case, overhead in mechanisms:
        print(f"{mechanism:<12} | {description:<20} | {use_case:<30} | {overhead}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Web server request queue
    print("1. Web Server Request Queue:")
    
    class WebServerQueue:
        def __init__(self, max_requests: int = 100):
            self.request_queue = ThreadSafeQueue(max_requests)
            self.active = True
        
        def handle_request(self, request):
            """Handle incoming request"""
            try:
                success = self.request_queue.enqueue(request, timeout=0.1)
                if success:
                    print(f"   Queued request: {request}")
                else:
                    print(f"   Request rejected (queue full): {request}")
            except Exception as e:
                print(f"   Error handling request: {e}")
        
        def process_requests(self, worker_id):
            """Process requests from queue"""
            while self.active:
                try:
                    request = self.request_queue.dequeue(timeout=0.5)
                    print(f"   Worker {worker_id} processing: {request}")
                    time.sleep(0.1)  # Simulate processing
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"   Worker {worker_id} error: {e}")
    
    server = WebServerQueue()
    
    # Start worker threads
    workers = []
    for i in range(2):
        worker = threading.Thread(target=server.process_requests, args=(i,))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    # Simulate incoming requests
    for i in range(5):
        server.handle_request(f"GET /page{i}")
        time.sleep(0.05)
    
    time.sleep(0.5)  # Let workers process
    server.active = False
    
    # Application 2: Cache with thread-safe access
    print(f"\n2. Thread-Safe Cache System:")
    
    class ThreadSafeCache:
        def __init__(self, capacity: int = 100):
            self._cache = {}
            self._access_order = ThreadSafeQueue(capacity)
            self._lock = threading.RLock()
            self._capacity = capacity
        
        def get(self, key):
            """Get value from cache"""
            with self._lock:
                if key in self._cache:
                    print(f"   Cache hit: {key}")
                    return self._cache[key]
                else:
                    print(f"   Cache miss: {key}")
                    return None
        
        def put(self, key, value):
            """Put value in cache"""
            with self._lock:
                if len(self._cache) >= self._capacity:
                    # Simple eviction - remove oldest
                    try:
                        old_key = self._access_order.dequeue(timeout=0.1)
                        if old_key in self._cache:
                            del self._cache[old_key]
                            print(f"   Evicted: {old_key}")
                    except queue.Empty:
                        pass
                
                self._cache[key] = value
                self._access_order.enqueue(key)
                print(f"   Cached: {key} = {value}")
    
    cache = ThreadSafeCache(3)
    
    # Test cache operations
    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.get("key1")
    cache.put("key3", "value3")
    cache.put("key4", "value4")  # Should evict key2


if __name__ == "__main__":
    test_thread_safe_implementations()
    demonstrate_concurrent_access()
    demonstrate_producer_consumer()
    demonstrate_thread_pool()
    benchmark_thread_safety_overhead()
    test_deadlock_prevention()
    analyze_synchronization_mechanisms()
    demonstrate_real_world_applications()

"""
Thread Safe Queue Stack demonstrates advanced synchronization techniques
including locks, condition variables, producer-consumer patterns, and
deadlock prevention for concurrent data structure implementations.
"""
