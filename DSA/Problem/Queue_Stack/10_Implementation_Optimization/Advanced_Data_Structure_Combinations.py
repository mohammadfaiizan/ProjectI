"""
Advanced Data Structure Combinations - Multiple Approaches
Difficulty: Hard

Implement advanced combinations of queue and stack data structures.
Focus on hybrid structures, multi-level systems, and complex interactions.
"""

from typing import Any, Optional, List, Dict, Tuple, Generic, TypeVar
from collections import deque, defaultdict
import heapq
import threading
import time
import random

T = TypeVar('T')

class StackQueue:
    """
    Approach 1: Stack-Queue Hybrid (Steque)
    
    Data structure that supports both stack and queue operations efficiently.
    
    Time: O(1) for all operations, Space: O(n)
    """
    
    def __init__(self):
        self._left_stack = []   # For queue front operations
        self._right_stack = []  # For stack/queue rear operations
    
    def push_back(self, item: Any) -> None:
        """Push to back (stack push / queue enqueue)"""
        self._right_stack.append(item)
    
    def push_front(self, item: Any) -> None:
        """Push to front (queue front insertion)"""
        self._left_stack.append(item)
    
    def pop_back(self) -> Any:
        """Pop from back (stack pop)"""
        if self._right_stack:
            return self._right_stack.pop()
        elif self._left_stack:
            # Transfer all but one from left to right
            while len(self._left_stack) > 1:
                self._right_stack.append(self._left_stack.pop())
            return self._left_stack.pop()
        else:
            raise IndexError("StackQueue is empty")
    
    def pop_front(self) -> Any:
        """Pop from front (queue dequeue)"""
        if self._left_stack:
            return self._left_stack.pop()
        elif self._right_stack:
            # Transfer all but one from right to left
            while len(self._right_stack) > 1:
                self._left_stack.append(self._right_stack.pop())
            return self._right_stack.pop()
        else:
            raise IndexError("StackQueue is empty")
    
    def peek_back(self) -> Any:
        """Peek at back element"""
        if self._right_stack:
            return self._right_stack[-1]
        elif self._left_stack:
            return self._left_stack[0]
        else:
            raise IndexError("StackQueue is empty")
    
    def peek_front(self) -> Any:
        """Peek at front element"""
        if self._left_stack:
            return self._left_stack[-1]
        elif self._right_stack:
            return self._right_stack[0]
        else:
            raise IndexError("StackQueue is empty")
    
    def is_empty(self) -> bool:
        """Check if empty"""
        return len(self._left_stack) == 0 and len(self._right_stack) == 0
    
    def size(self) -> int:
        """Get total size"""
        return len(self._left_stack) + len(self._right_stack)


class MultiLevelQueue:
    """
    Approach 2: Multi-Level Priority Queue System
    
    Queue system with multiple priority levels and aging mechanism.
    
    Time: O(log n) for priority operations, O(1) for level operations
    """
    
    def __init__(self, num_levels: int = 5, aging_threshold: int = 100):
        self.num_levels = num_levels
        self.aging_threshold = aging_threshold
        self.queues = [deque() for _ in range(num_levels)]
        self.item_ages = {}  # item_id -> age
        self.next_id = 0
        self.total_operations = 0
    
    def enqueue(self, item: Any, priority: int = 0) -> int:
        """Enqueue item with priority level"""
        priority = max(0, min(priority, self.num_levels - 1))
        item_id = self.next_id
        self.next_id += 1
        
        self.queues[priority].append((item, item_id))
        self.item_ages[item_id] = 0
        
        return item_id
    
    def dequeue(self) -> Tuple[Any, int]:
        """Dequeue highest priority item"""
        self.total_operations += 1
        
        # Age items periodically
        if self.total_operations % 10 == 0:
            self._age_items()
        
        # Find highest priority non-empty queue
        for level in range(self.num_levels):
            if self.queues[level]:
                item, item_id = self.queues[level].popleft()
                del self.item_ages[item_id]
                return item, level
        
        raise IndexError("All queues are empty")
    
    def _age_items(self) -> None:
        """Age items and promote them if necessary"""
        for item_id in list(self.item_ages.keys()):
            self.item_ages[item_id] += 1
            
            if self.item_ages[item_id] >= self.aging_threshold:
                # Find and promote the item
                self._promote_item(item_id)
    
    def _promote_item(self, item_id: int) -> None:
        """Promote aged item to higher priority level"""
        for level in range(self.num_levels - 1, 0, -1):
            queue = self.queues[level]
            for i, (item, iid) in enumerate(queue):
                if iid == item_id:
                    # Remove from current level
                    del queue[i]
                    # Add to higher priority level
                    self.queues[level - 1].append((item, item_id))
                    self.item_ages[item_id] = 0
                    return
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            'level_sizes': [len(q) for q in self.queues],
            'total_items': sum(len(q) for q in self.queues),
            'aged_items': sum(1 for age in self.item_ages.values() if age > 0),
            'operations': self.total_operations
        }


class AdaptiveStack:
    """
    Approach 3: Adaptive Stack with Multiple Backends
    
    Stack that adapts its internal representation based on usage patterns.
    
    Time: O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self._items = []
        self._operation_count = 0
        self._push_count = 0
        self._pop_count = 0
        self._backend = "list"  # "list", "deque", "linked"
        self._adaptation_threshold = 1000
    
    def push(self, item: Any) -> None:
        """Adaptive push operation"""
        self._operation_count += 1
        self._push_count += 1
        
        if self._backend == "list":
            self._items.append(item)
        elif self._backend == "deque":
            self._items.append(item)
        elif self._backend == "linked":
            new_node = {'data': item, 'next': getattr(self, '_top', None)}
            self._top = new_node
        
        # Check if adaptation is needed
        if self._operation_count % self._adaptation_threshold == 0:
            self._adapt_backend()
    
    def pop(self) -> Any:
        """Adaptive pop operation"""
        self._operation_count += 1
        self._pop_count += 1
        
        if self._backend == "list":
            if not self._items:
                raise IndexError("Stack underflow")
            return self._items.pop()
        elif self._backend == "deque":
            if not self._items:
                raise IndexError("Stack underflow")
            return self._items.pop()
        elif self._backend == "linked":
            if not hasattr(self, '_top') or self._top is None:
                raise IndexError("Stack underflow")
            data = self._top['data']
            self._top = self._top['next']
            return data
    
    def _adapt_backend(self) -> None:
        """Adapt backend based on usage patterns"""
        push_ratio = self._push_count / self._operation_count
        
        # Convert current items to list for migration
        current_items = []
        try:
            while True:
                current_items.append(self.pop())
        except IndexError:
            pass
        
        # Choose optimal backend
        if push_ratio > 0.8:  # Push-heavy workload
            self._switch_to_deque()
        elif push_ratio < 0.3:  # Pop-heavy workload
            self._switch_to_linked()
        else:  # Balanced workload
            self._switch_to_list()
        
        # Restore items
        for item in reversed(current_items):
            self.push(item)
    
    def _switch_to_list(self) -> None:
        """Switch to list backend"""
        self._backend = "list"
        self._items = []
    
    def _switch_to_deque(self) -> None:
        """Switch to deque backend"""
        self._backend = "deque"
        self._items = deque()
    
    def _switch_to_linked(self) -> None:
        """Switch to linked list backend"""
        self._backend = "linked"
        self._top = None
    
    def get_stats(self) -> Dict:
        """Get adaptation statistics"""
        return {
            'backend': self._backend,
            'operations': self._operation_count,
            'push_ratio': self._push_count / max(1, self._operation_count),
            'pop_ratio': self._pop_count / max(1, self._operation_count)
        }


class HybridPriorityQueue:
    """
    Approach 4: Hybrid Priority Queue with Multiple Strategies
    
    Priority queue that combines heap, sorted list, and bucketing strategies.
    
    Time: Varies by strategy, Space: O(n)
    """
    
    def __init__(self, strategy: str = "auto"):
        self.strategy = strategy
        self.heap = []
        self.sorted_list = []
        self.buckets = defaultdict(list)
        self.size = 0
        self.operation_count = 0
        
        # Performance tracking
        self.heap_time = 0
        self.sorted_time = 0
        self.bucket_time = 0
    
    def enqueue(self, item: Any, priority: int) -> None:
        """Enqueue with adaptive strategy"""
        self.operation_count += 1
        self.size += 1
        
        if self.strategy == "heap" or (self.strategy == "auto" and self.size > 100):
            start_time = time.time()
            heapq.heappush(self.heap, (priority, item))
            self.heap_time += time.time() - start_time
            
        elif self.strategy == "sorted" or (self.strategy == "auto" and self.size <= 20):
            start_time = time.time()
            # Binary search insertion
            import bisect
            bisect.insort(self.sorted_list, (priority, item))
            self.sorted_time += time.time() - start_time
            
        elif self.strategy == "bucket" or (self.strategy == "auto" and 20 < self.size <= 100):
            start_time = time.time()
            self.buckets[priority].append(item)
            self.bucket_time += time.time() - start_time
    
    def dequeue(self) -> Tuple[Any, int]:
        """Dequeue with adaptive strategy"""
        if self.size == 0:
            raise IndexError("Priority queue is empty")
        
        self.size -= 1
        
        if self.strategy == "heap" or (self.strategy == "auto" and self.heap):
            start_time = time.time()
            priority, item = heapq.heappop(self.heap)
            self.heap_time += time.time() - start_time
            return item, priority
            
        elif self.strategy == "sorted" or (self.strategy == "auto" and self.sorted_list):
            start_time = time.time()
            priority, item = self.sorted_list.pop(0)
            self.sorted_time += time.time() - start_time
            return item, priority
            
        elif self.strategy == "bucket" or (self.strategy == "auto" and self.buckets):
            start_time = time.time()
            min_priority = min(self.buckets.keys())
            item = self.buckets[min_priority].pop(0)
            if not self.buckets[min_priority]:
                del self.buckets[min_priority]
            self.bucket_time += time.time() - start_time
            return item, min_priority
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for different strategies"""
        total_time = self.heap_time + self.sorted_time + self.bucket_time
        return {
            'heap_time': self.heap_time,
            'sorted_time': self.sorted_time,
            'bucket_time': self.bucket_time,
            'total_time': total_time,
            'heap_percentage': (self.heap_time / max(total_time, 1e-9)) * 100,
            'current_strategy': self.strategy,
            'size': self.size
        }


class ConcurrentStackQueue:
    """
    Approach 5: Lock-Free Concurrent Stack-Queue
    
    Thread-safe combination structure using atomic operations.
    
    Time: O(1) for operations, Space: O(n)
    """
    
    def __init__(self):
        self._stack_items = []
        self._queue_items = deque()
        self._stack_lock = threading.RLock()
        self._queue_lock = threading.RLock()
        self._mode_lock = threading.RLock()
        self._mode = "balanced"  # "stack_heavy", "queue_heavy", "balanced"
        self._operation_counts = {"stack": 0, "queue": 0}
    
    def push(self, item: Any) -> None:
        """Thread-safe stack push"""
        with self._stack_lock:
            self._stack_items.append(item)
        
        with self._mode_lock:
            self._operation_counts["stack"] += 1
            self._update_mode()
    
    def pop(self) -> Any:
        """Thread-safe stack pop"""
        with self._stack_lock:
            if not self._stack_items:
                raise IndexError("Stack is empty")
            return self._stack_items.pop()
    
    def enqueue(self, item: Any) -> None:
        """Thread-safe queue enqueue"""
        with self._queue_lock:
            self._queue_items.append(item)
        
        with self._mode_lock:
            self._operation_counts["queue"] += 1
            self._update_mode()
    
    def dequeue(self) -> Any:
        """Thread-safe queue dequeue"""
        with self._queue_lock:
            if not self._queue_items:
                raise IndexError("Queue is empty")
            return self._queue_items.popleft()
    
    def smart_get(self) -> Tuple[Any, str]:
        """Get item using adaptive strategy"""
        with self._mode_lock:
            mode = self._mode
        
        if mode == "stack_heavy":
            try:
                return self.pop(), "stack"
            except IndexError:
                return self.dequeue(), "queue"
        elif mode == "queue_heavy":
            try:
                return self.dequeue(), "queue"
            except IndexError:
                return self.pop(), "stack"
        else:  # balanced
            # Alternate between stack and queue
            if random.random() < 0.5:
                try:
                    return self.pop(), "stack"
                except IndexError:
                    return self.dequeue(), "queue"
            else:
                try:
                    return self.dequeue(), "queue"
                except IndexError:
                    return self.pop(), "stack"
    
    def _update_mode(self) -> None:
        """Update operation mode based on usage patterns"""
        total_ops = sum(self._operation_counts.values())
        if total_ops < 100:
            return
        
        stack_ratio = self._operation_counts["stack"] / total_ops
        
        if stack_ratio > 0.7:
            self._mode = "stack_heavy"
        elif stack_ratio < 0.3:
            self._mode = "queue_heavy"
        else:
            self._mode = "balanced"
    
    def get_stats(self) -> Dict:
        """Get concurrent operation statistics"""
        with self._mode_lock:
            return {
                'mode': self._mode,
                'stack_operations': self._operation_counts["stack"],
                'queue_operations': self._operation_counts["queue"],
                'stack_size': len(self._stack_items),
                'queue_size': len(self._queue_items)
            }


class TimeSensitiveQueue:
    """
    Approach 6: Time-Sensitive Queue with TTL
    
    Queue that automatically expires items based on time-to-live.
    
    Time: O(log n) for operations with cleanup, Space: O(n)
    """
    
    def __init__(self, default_ttl: float = 60.0):
        self.default_ttl = default_ttl
        self.items = []  # (timestamp, expiry_time, item)
        self.expiry_heap = []  # (expiry_time, timestamp, item)
        self.next_timestamp = 0
    
    def enqueue(self, item: Any, ttl: Optional[float] = None) -> None:
        """Enqueue item with TTL"""
        current_time = time.time()
        ttl = ttl or self.default_ttl
        expiry_time = current_time + ttl
        timestamp = self.next_timestamp
        self.next_timestamp += 1
        
        self.items.append((timestamp, expiry_time, item))
        heapq.heappush(self.expiry_heap, (expiry_time, timestamp, item))
        
        # Cleanup expired items periodically
        if len(self.items) % 10 == 0:
            self._cleanup_expired()
    
    def dequeue(self) -> Any:
        """Dequeue non-expired item"""
        self._cleanup_expired()
        
        current_time = time.time()
        
        # Find first non-expired item
        while self.items:
            timestamp, expiry_time, item = self.items[0]
            
            if expiry_time > current_time:
                # Item is still valid
                self.items.pop(0)
                return item
            else:
                # Item expired, remove it
                self.items.pop(0)
        
        raise IndexError("No valid items in queue")
    
    def _cleanup_expired(self) -> None:
        """Remove expired items from heap"""
        current_time = time.time()
        
        while self.expiry_heap and self.expiry_heap[0][0] <= current_time:
            heapq.heappop(self.expiry_heap)
        
        # Also clean up items list
        self.items = [(ts, exp, item) for ts, exp, item in self.items 
                     if exp > current_time]
    
    def get_valid_count(self) -> int:
        """Get count of non-expired items"""
        self._cleanup_expired()
        return len(self.items)
    
    def get_expired_count(self) -> int:
        """Get count of expired items that were cleaned up"""
        total_added = self.next_timestamp
        current_valid = self.get_valid_count()
        return total_added - current_valid


def test_advanced_combinations():
    """Test advanced data structure combinations"""
    print("=== Testing Advanced Data Structure Combinations ===")
    
    # Test StackQueue
    print("\n--- Stack-Queue Hybrid ---")
    steque = StackQueue()
    
    # Test mixed operations
    steque.push_back(1)
    steque.push_back(2)
    steque.push_front(0)
    
    print(f"After mixed pushes: size = {steque.size()}")
    print(f"Front: {steque.peek_front()}, Back: {steque.peek_back()}")
    
    print(f"Pop front: {steque.pop_front()}")
    print(f"Pop back: {steque.pop_back()}")
    
    # Test MultiLevelQueue
    print("\n--- Multi-Level Priority Queue ---")
    mlq = MultiLevelQueue(num_levels=3, aging_threshold=5)
    
    # Add items with different priorities
    mlq.enqueue("Low priority", 2)
    mlq.enqueue("High priority", 0)
    mlq.enqueue("Medium priority", 1)
    mlq.enqueue("Another low", 2)
    
    print("Initial stats:", mlq.get_stats())
    
    # Process some items
    for _ in range(2):
        item, level = mlq.dequeue()
        print(f"Dequeued: {item} from level {level}")
    
    print("After dequeue:", mlq.get_stats())
    
    # Test AdaptiveStack
    print("\n--- Adaptive Stack ---")
    adaptive_stack = AdaptiveStack()
    
    # Simulate push-heavy workload
    for i in range(50):
        adaptive_stack.push(i)
    
    print("After push-heavy workload:", adaptive_stack.get_stats())
    
    # Simulate pop-heavy workload
    for _ in range(30):
        adaptive_stack.pop()
    
    print("After pop-heavy workload:", adaptive_stack.get_stats())


def demonstrate_concurrent_operations():
    """Demonstrate concurrent stack-queue operations"""
    print("\n=== Concurrent Operations Demonstration ===")
    
    concurrent_sq = ConcurrentStackQueue()
    results = []
    
    def stack_worker(worker_id: int):
        """Worker that uses stack operations"""
        for i in range(10):
            concurrent_sq.push(f"S{worker_id}_{i}")
            time.sleep(0.001)
        
        for _ in range(5):
            try:
                item = concurrent_sq.pop()
                results.append(f"Worker {worker_id} popped: {item}")
            except IndexError:
                results.append(f"Worker {worker_id} found empty stack")
            time.sleep(0.001)
    
    def queue_worker(worker_id: int):
        """Worker that uses queue operations"""
        for i in range(10):
            concurrent_sq.enqueue(f"Q{worker_id}_{i}")
            time.sleep(0.001)
        
        for _ in range(5):
            try:
                item = concurrent_sq.dequeue()
                results.append(f"Worker {worker_id} dequeued: {item}")
            except IndexError:
                results.append(f"Worker {worker_id} found empty queue")
            time.sleep(0.001)
    
    def smart_worker(worker_id: int):
        """Worker that uses smart_get"""
        for _ in range(8):
            try:
                item, source = concurrent_sq.smart_get()
                results.append(f"Smart worker {worker_id} got {item} from {source}")
            except IndexError:
                results.append(f"Smart worker {worker_id} found everything empty")
            time.sleep(0.001)
    
    # Start workers
    threads = []
    
    # Stack workers
    for i in range(2):
        t = threading.Thread(target=stack_worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Queue workers
    for i in range(2, 4):
        t = threading.Thread(target=queue_worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Smart workers
    for i in range(4, 6):
        t = threading.Thread(target=smart_worker, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for completion
    for t in threads:
        t.join()
    
    print("Concurrent operations results:")
    for result in results[:15]:  # Show first 15 results
        print(f"  {result}")
    
    print(f"Final stats: {concurrent_sq.get_stats()}")


def demonstrate_time_sensitive_queue():
    """Demonstrate time-sensitive queue with TTL"""
    print("\n=== Time-Sensitive Queue Demonstration ===")
    
    ttl_queue = TimeSensitiveQueue(default_ttl=0.1)  # 100ms TTL
    
    # Add items with different TTLs
    ttl_queue.enqueue("Short lived", 0.05)  # 50ms
    ttl_queue.enqueue("Medium lived", 0.1)  # 100ms
    ttl_queue.enqueue("Long lived", 0.2)    # 200ms
    
    print(f"Added 3 items, valid count: {ttl_queue.get_valid_count()}")
    
    # Wait a bit
    time.sleep(0.06)
    print(f"After 60ms, valid count: {ttl_queue.get_valid_count()}")
    
    # Try to dequeue
    try:
        item = ttl_queue.dequeue()
        print(f"Dequeued: {item}")
    except IndexError as e:
        print(f"Dequeue failed: {e}")
    
    print(f"Valid count: {ttl_queue.get_valid_count()}")
    print(f"Expired count: {ttl_queue.get_expired_count()}")
    
    # Wait more
    time.sleep(0.15)
    print(f"After total 210ms, valid count: {ttl_queue.get_valid_count()}")


def benchmark_hybrid_strategies():
    """Benchmark different hybrid priority queue strategies"""
    print("\n=== Hybrid Priority Queue Strategy Benchmark ===")
    
    strategies = ["heap", "sorted", "bucket", "auto"]
    n_items = 1000
    
    for strategy in strategies:
        pq = HybridPriorityQueue(strategy=strategy)
        
        # Add items
        start_time = time.time()
        for i in range(n_items):
            priority = random.randint(0, 100)
            pq.enqueue(f"item_{i}", priority)
        
        # Remove items
        for _ in range(n_items):
            pq.dequeue()
        
        total_time = time.time() - start_time
        stats = pq.get_performance_stats()
        
        print(f"Strategy: {strategy:10} | Total time: {total_time:.4f}s | Stats: {stats}")


def analyze_combination_benefits():
    """Analyze benefits of data structure combinations"""
    print("\n=== Data Structure Combination Benefits ===")
    
    benefits = [
        ("StackQueue", "Unified interface", "Both LIFO and FIFO in one structure", "Versatile applications"),
        ("MultiLevel Queue", "Priority with aging", "Prevents starvation", "Fair scheduling systems"),
        ("Adaptive Stack", "Performance optimization", "Adapts to usage patterns", "Dynamic workloads"),
        ("Hybrid Priority Queue", "Strategy selection", "Optimal for different sizes", "Varying data sizes"),
        ("Concurrent StackQueue", "Thread safety", "Lock-based synchronization", "Multi-threaded applications"),
        ("Time-Sensitive Queue", "Automatic cleanup", "TTL-based expiration", "Cache-like behavior"),
    ]
    
    print(f"{'Structure':<20} | {'Key Feature':<20} | {'Benefit':<30} | {'Use Case'}")
    print("-" * 100)
    
    for structure, feature, benefit, use_case in benefits:
        print(f"{structure:<20} | {feature:<20} | {benefit:<30} | {use_case}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Task scheduling system
    print("1. Advanced Task Scheduling System:")
    
    class TaskScheduler:
        def __init__(self):
            self.priority_queue = MultiLevelQueue(num_levels=4, aging_threshold=50)
            self.task_counter = 0
        
        def submit_task(self, task_name: str, priority: int, task_type: str):
            task_id = self.priority_queue.enqueue({
                'name': task_name,
                'type': task_type,
                'submitted_at': time.time()
            }, priority)
            self.task_counter += 1
            print(f"   Submitted: {task_name} (priority {priority})")
            return task_id
        
        def execute_next_task(self):
            try:
                task, priority = self.priority_queue.dequeue()
                print(f"   Executing: {task['name']} from priority level {priority}")
                return task
            except IndexError:
                print("   No tasks to execute")
                return None
        
        def get_system_stats(self):
            return self.priority_queue.get_stats()
    
    scheduler = TaskScheduler()
    
    # Submit various tasks
    scheduler.submit_task("Database backup", 2, "maintenance")
    scheduler.submit_task("User request", 0, "interactive")
    scheduler.submit_task("Log rotation", 3, "maintenance")
    scheduler.submit_task("Email sending", 1, "background")
    scheduler.submit_task("Critical alert", 0, "urgent")
    
    print(f"   System stats: {scheduler.get_system_stats()}")
    
    # Execute some tasks
    for _ in range(3):
        scheduler.execute_next_task()
    
    # Application 2: Cache system with multiple eviction strategies
    print(f"\n2. Multi-Strategy Cache System:")
    
    class MultiStrategyCache:
        def __init__(self, capacity: int = 100):
            self.capacity = capacity
            self.lru_stack = StackQueue()  # For LRU tracking
            self.priority_queue = HybridPriorityQueue("auto")  # For priority-based eviction
            self.ttl_queue = TimeSensitiveQueue(default_ttl=300)  # 5 minute TTL
            self.cache_data = {}
            self.access_counts = defaultdict(int)
        
        def put(self, key: str, value: Any, priority: int = 0, ttl: Optional[float] = None):
            # Store in multiple structures for different eviction strategies
            self.cache_data[key] = value
            self.lru_stack.push_back(key)  # Most recent at back
            self.priority_queue.enqueue(key, priority)
            self.ttl_queue.enqueue(key, ttl)
            
            print(f"   Cached: {key} with priority {priority}")
            
            # Evict if over capacity (simplified)
            if len(self.cache_data) > self.capacity:
                self._evict_item()
        
        def get(self, key: str) -> Optional[Any]:
            if key in self.cache_data:
                self.access_counts[key] += 1
                # Update LRU position
                self.lru_stack.push_back(key)
                print(f"   Cache hit: {key}")
                return self.cache_data[key]
            else:
                print(f"   Cache miss: {key}")
                return None
        
        def _evict_item(self):
            # Use LRU strategy for this example
            try:
                lru_key = self.lru_stack.pop_front()
                if lru_key in self.cache_data:
                    del self.cache_data[lru_key]
                    print(f"   Evicted: {lru_key}")
            except IndexError:
                pass
        
        def cleanup_expired(self):
            # Clean up TTL expired items
            try:
                while True:
                    expired_key = self.ttl_queue.dequeue()
                    if expired_key in self.cache_data:
                        del self.cache_data[expired_key]
                        print(f"   TTL expired: {expired_key}")
            except IndexError:
                pass  # No more expired items
    
    cache = MultiStrategyCache(capacity=3)
    
    # Test cache operations
    cache.put("user:123", {"name": "Alice"}, priority=1)
    cache.put("session:abc", {"token": "xyz"}, priority=0, ttl=0.1)
    cache.put("config:app", {"theme": "dark"}, priority=2)
    
    # Access some items
    cache.get("user:123")
    cache.get("nonexistent")
    
    # Add more items to trigger eviction
    cache.put("temp:data", {"value": 42}, priority=0)
    
    # Wait and cleanup expired items
    time.sleep(0.11)
    cache.cleanup_expired()


if __name__ == "__main__":
    test_advanced_combinations()
    demonstrate_concurrent_operations()
    demonstrate_time_sensitive_queue()
    benchmark_hybrid_strategies()
    analyze_combination_benefits()
    demonstrate_real_world_applications()

"""
Advanced Data Structure Combinations demonstrates sophisticated hybrid
structures including stack-queue combinations, multi-level priority systems,
adaptive backends, concurrent access patterns, and time-sensitive operations.
"""
