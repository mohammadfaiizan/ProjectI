"""
1172. Dinner Plate Stacks - Multiple Approaches
Difficulty: Hard

You have an infinite number of stacks arranged in a row and numbered (left to right) from 0, 1, 2, ..., each with the same maximum capacity.

Implement the DinnerPlates class:
- DinnerPlates(int capacity) Initializes the object with the maximum capacity of the stacks capacity.
- void push(int val) Pushes the given integer val into the leftmost stack with available space.
- int pop() Pops and returns the top element of the rightmost non-empty stack. If no non-empty stack exists, return -1.
- int popAtStack(int index) Pops and returns the top element of the stack with the given index index. If the stack with the given index is empty, return -1.
"""

from typing import List, Dict
import heapq
from collections import deque

class DinnerPlatesSimple:
    """
    Approach 1: Simple List of Stacks
    
    Use list of stacks and linear search for operations.
    
    Time Complexity:
    - push: O(n) to find leftmost available stack
    - pop: O(n) to find rightmost non-empty stack
    - popAtStack: O(1)
    
    Space Complexity: O(n * capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []
    
    def push(self, val: int) -> None:
        # Find leftmost stack with available space
        for i, stack in enumerate(self.stacks):
            if len(stack) < self.capacity:
                stack.append(val)
                return
        
        # No available stack, create new one
        self.stacks.append([val])
    
    def pop(self) -> int:
        # Find rightmost non-empty stack
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()  # Remove empty stacks
        
        if not self.stacks:
            return -1
        
        return self.stacks[-1].pop()
    
    def popAtStack(self, index: int) -> int:
        if index >= len(self.stacks) or not self.stacks[index]:
            return -1
        
        return self.stacks[index].pop()

class DinnerPlatesOptimal:
    """
    Approach 2: Optimal with Min-Heap
    
    Use min-heap to track available stacks efficiently.
    
    Time Complexity:
    - push: O(log n)
    - pop: O(1) amortized
    - popAtStack: O(log n)
    
    Space Complexity: O(n * capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []
        self.available_stacks = []  # Min-heap of indices with available space
    
    def push(self, val: int) -> None:
        # Clean up available_stacks heap
        while self.available_stacks and (
            self.available_stacks[0] >= len(self.stacks) or 
            len(self.stacks[self.available_stacks[0]]) >= self.capacity
        ):
            heapq.heappop(self.available_stacks)
        
        if self.available_stacks:
            # Use leftmost available stack
            index = self.available_stacks[0]
            self.stacks[index].append(val)
            
            # If stack becomes full, remove from available
            if len(self.stacks[index]) >= self.capacity:
                heapq.heappop(self.available_stacks)
        else:
            # Create new stack
            self.stacks.append([val])
            
            # If not full, add to available stacks
            if self.capacity > 1:
                heapq.heappush(self.available_stacks, len(self.stacks) - 1)
    
    def pop(self) -> int:
        # Remove empty stacks from the right
        while self.stacks and not self.stacks[-1]:
            self.stacks.pop()
        
        if not self.stacks:
            return -1
        
        val = self.stacks[-1].pop()
        
        # Add back to available if not full and not already there
        last_index = len(self.stacks) - 1
        if (len(self.stacks[-1]) < self.capacity and 
            last_index not in self.available_stacks):
            heapq.heappush(self.available_stacks, last_index)
        
        return val
    
    def popAtStack(self, index: int) -> int:
        if index >= len(self.stacks) or not self.stacks[index]:
            return -1
        
        val = self.stacks[index].pop()
        
        # Add to available stacks if has space
        if len(self.stacks[index]) < self.capacity:
            heapq.heappush(self.available_stacks, index)
        
        return val

class DinnerPlatesAdvanced:
    """
    Approach 3: Advanced with Multiple Data Structures
    
    Use multiple data structures for optimal performance.
    
    Time Complexity:
    - push: O(log n)
    - pop: O(1) amortized
    - popAtStack: O(log n)
    
    Space Complexity: O(n * capacity)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = []
        self.available_stacks = []  # Min-heap for available stacks
        self.rightmost_non_empty = -1  # Cache rightmost non-empty stack
        
        # Statistics
        self.total_pushes = 0
        self.total_pops = 0
        self.total_pop_at_stack = 0
        self.max_stacks_used = 0
    
    def push(self, val: int) -> None:
        self.total_pushes += 1
        
        # Clean up available_stacks
        while self.available_stacks and (
            self.available_stacks[0] >= len(self.stacks) or 
            len(self.stacks[self.available_stacks[0]]) >= self.capacity
        ):
            heapq.heappop(self.available_stacks)
        
        if self.available_stacks:
            # Use leftmost available stack
            index = self.available_stacks[0]
            self.stacks[index].append(val)
            
            # Update rightmost non-empty
            self.rightmost_non_empty = max(self.rightmost_non_empty, index)
            
            # Remove from available if full
            if len(self.stacks[index]) >= self.capacity:
                heapq.heappop(self.available_stacks)
        else:
            # Create new stack
            self.stacks.append([val])
            new_index = len(self.stacks) - 1
            
            # Update rightmost non-empty
            self.rightmost_non_empty = new_index
            
            # Add to available if not full
            if self.capacity > 1:
                heapq.heappush(self.available_stacks, new_index)
        
        # Update statistics
        self.max_stacks_used = max(self.max_stacks_used, len(self.stacks))
    
    def pop(self) -> int:
        self.total_pops += 1
        
        # Update rightmost non-empty
        while self.rightmost_non_empty >= 0 and (
            self.rightmost_non_empty >= len(self.stacks) or 
            not self.stacks[self.rightmost_non_empty]
        ):
            self.rightmost_non_empty -= 1
        
        if self.rightmost_non_empty < 0:
            return -1
        
        val = self.stacks[self.rightmost_non_empty].pop()
        
        # Add to available stacks if has space
        if len(self.stacks[self.rightmost_non_empty]) < self.capacity:
            heapq.heappush(self.available_stacks, self.rightmost_non_empty)
        
        return val
    
    def popAtStack(self, index: int) -> int:
        self.total_pop_at_stack += 1
        
        if index >= len(self.stacks) or not self.stacks[index]:
            return -1
        
        val = self.stacks[index].pop()
        
        # Add to available stacks if has space
        if len(self.stacks[index]) < self.capacity:
            heapq.heappush(self.available_stacks, index)
        
        # Update rightmost non-empty if necessary
        if index == self.rightmost_non_empty and not self.stacks[index]:
            while self.rightmost_non_empty >= 0 and (
                self.rightmost_non_empty >= len(self.stacks) or 
                not self.stacks[self.rightmost_non_empty]
            ):
                self.rightmost_non_empty -= 1
        
        return val
    
    def get_statistics(self) -> dict:
        """Get operation statistics"""
        active_stacks = sum(1 for stack in self.stacks if stack)
        total_elements = sum(len(stack) for stack in self.stacks)
        
        return {
            'total_pushes': self.total_pushes,
            'total_pops': self.total_pops,
            'total_pop_at_stack': self.total_pop_at_stack,
            'max_stacks_used': self.max_stacks_used,
            'current_active_stacks': active_stacks,
            'total_elements': total_elements,
            'average_stack_utilization': total_elements / max(1, len(self.stacks) * self.capacity)
        }
    
    def get_state(self) -> dict:
        """Get current state for debugging"""
        return {
            'stacks': [len(stack) for stack in self.stacks],
            'available_stacks': list(self.available_stacks),
            'rightmost_non_empty': self.rightmost_non_empty,
            'total_stacks': len(self.stacks)
        }

class DinnerPlatesMemoryOptimized:
    """
    Approach 4: Memory-Optimized Implementation
    
    Optimize memory usage by removing empty stacks and compacting.
    
    Time Complexity:
    - push: O(log n)
    - pop: O(1) amortized
    - popAtStack: O(log n)
    
    Space Complexity: O(active_elements)
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stacks = {}  # Sparse representation: index -> stack
        self.available_stacks = []  # Min-heap
        self.max_index = -1
        
        # Memory management
        self.cleanup_threshold = 100  # Clean up every 100 operations
        self.operations_count = 0
    
    def _cleanup_empty_stacks(self) -> None:
        """Remove empty stacks to save memory"""
        empty_indices = [idx for idx, stack in self.stacks.items() if not stack]
        
        for idx in empty_indices:
            del self.stacks[idx]
        
        # Update max_index
        if self.stacks:
            self.max_index = max(self.stacks.keys())
        else:
            self.max_index = -1
        
        # Clean up available_stacks
        self.available_stacks = [
            idx for idx in self.available_stacks 
            if idx in self.stacks and len(self.stacks[idx]) < self.capacity
        ]
        heapq.heapify(self.available_stacks)
    
    def push(self, val: int) -> None:
        self.operations_count += 1
        
        # Clean up available_stacks
        while self.available_stacks and (
            self.available_stacks[0] not in self.stacks or
            len(self.stacks[self.available_stacks[0]]) >= self.capacity
        ):
            heapq.heappop(self.available_stacks)
        
        if self.available_stacks:
            # Use leftmost available stack
            index = self.available_stacks[0]
            self.stacks[index].append(val)
            
            # Remove from available if full
            if len(self.stacks[index]) >= self.capacity:
                heapq.heappop(self.available_stacks)
        else:
            # Create new stack at leftmost available position
            new_index = 0
            while new_index in self.stacks:
                new_index += 1
            
            self.stacks[new_index] = [val]
            self.max_index = max(self.max_index, new_index)
            
            # Add to available if not full
            if self.capacity > 1:
                heapq.heappush(self.available_stacks, new_index)
        
        # Periodic cleanup
        if self.operations_count % self.cleanup_threshold == 0:
            self._cleanup_empty_stacks()
    
    def pop(self) -> int:
        self.operations_count += 1
        
        # Find rightmost non-empty stack
        while self.max_index >= 0 and (
            self.max_index not in self.stacks or not self.stacks[self.max_index]
        ):
            self.max_index -= 1
        
        if self.max_index < 0:
            return -1
        
        val = self.stacks[self.max_index].pop()
        
        # Add to available if has space
        if len(self.stacks[self.max_index]) < self.capacity:
            heapq.heappush(self.available_stacks, self.max_index)
        
        return val
    
    def popAtStack(self, index: int) -> int:
        self.operations_count += 1
        
        if index not in self.stacks or not self.stacks[index]:
            return -1
        
        val = self.stacks[index].pop()
        
        # Add to available if has space
        if len(self.stacks[index]) < self.capacity:
            heapq.heappush(self.available_stacks, index)
        
        return val
    
    def get_memory_stats(self) -> dict:
        """Get memory usage statistics"""
        active_stacks = len([stack for stack in self.stacks.values() if stack])
        total_elements = sum(len(stack) for stack in self.stacks.values())
        
        return {
            'total_stacks': len(self.stacks),
            'active_stacks': active_stacks,
            'total_elements': total_elements,
            'max_index': self.max_index,
            'available_stacks_count': len(self.available_stacks),
            'memory_efficiency': active_stacks / max(1, len(self.stacks))
        }

class DinnerPlatesConcurrent:
    """
    Approach 5: Thread-Safe Implementation
    
    Thread-safe version for concurrent access.
    
    Time Complexity: Same as optimal + lock overhead
    Space Complexity: O(n * capacity)
    """
    
    def __init__(self, capacity: int):
        import threading
        
        self.capacity = capacity
        self.stacks = []
        self.available_stacks = []
        
        # Thread safety
        self.lock = threading.RLock()
        self.operation_count = 0
    
    def push(self, val: int) -> None:
        with self.lock:
            self.operation_count += 1
            
            # Clean up available_stacks
            while self.available_stacks and (
                self.available_stacks[0] >= len(self.stacks) or 
                len(self.stacks[self.available_stacks[0]]) >= self.capacity
            ):
                heapq.heappop(self.available_stacks)
            
            if self.available_stacks:
                index = self.available_stacks[0]
                self.stacks[index].append(val)
                
                if len(self.stacks[index]) >= self.capacity:
                    heapq.heappop(self.available_stacks)
            else:
                self.stacks.append([val])
                
                if self.capacity > 1:
                    heapq.heappush(self.available_stacks, len(self.stacks) - 1)
    
    def pop(self) -> int:
        with self.lock:
            self.operation_count += 1
            
            while self.stacks and not self.stacks[-1]:
                self.stacks.pop()
            
            if not self.stacks:
                return -1
            
            val = self.stacks[-1].pop()
            
            last_index = len(self.stacks) - 1
            if len(self.stacks[-1]) < self.capacity:
                heapq.heappush(self.available_stacks, last_index)
            
            return val
    
    def popAtStack(self, index: int) -> int:
        with self.lock:
            self.operation_count += 1
            
            if index >= len(self.stacks) or not self.stacks[index]:
                return -1
            
            val = self.stacks[index].pop()
            
            if len(self.stacks[index]) < self.capacity:
                heapq.heappush(self.available_stacks, index)
            
            return val
    
    def get_stats(self) -> dict:
        """Get thread-safe statistics"""
        with self.lock:
            return {
                'operation_count': self.operation_count,
                'total_stacks': len(self.stacks),
                'available_stacks': len(self.available_stacks)
            }


def test_dinner_plates_basic():
    """Test basic DinnerPlates functionality"""
    print("=== Testing Basic DinnerPlates Functionality ===")
    
    implementations = [
        ("Simple", DinnerPlatesSimple),
        ("Optimal", DinnerPlatesOptimal),
        ("Advanced", DinnerPlatesAdvanced),
        ("Memory Optimized", DinnerPlatesMemoryOptimized),
        ("Concurrent", DinnerPlatesConcurrent)
    ]
    
    for name, DinnerPlatesClass in implementations:
        print(f"\n{name}:")
        
        dp = DinnerPlatesClass(2)  # Capacity 2
        
        # Test sequence from problem
        operations = [
            ("push", 1), ("push", 2), ("push", 3), ("push", 4), ("push", 5),
            ("popAtStack", 0), ("push", 20), ("push", 21), ("popAtStack", 0),
            ("popAtStack", 2), ("pop"), ("pop"), ("pop"), ("pop"), ("pop")
        ]
        
        for op, *args in operations:
            if op == "push":
                dp.push(args[0])
                print(f"  push({args[0]})")
            elif op == "pop":
                result = dp.pop()
                print(f"  pop(): {result}")
            elif op == "popAtStack":
                result = dp.popAtStack(args[0])
                print(f"  popAtStack({args[0]}): {result}")

def test_dinner_plates_edge_cases():
    """Test edge cases"""
    print("\n=== Testing DinnerPlates Edge Cases ===")
    
    # Test with capacity 1
    print("Capacity 1:")
    dp = DinnerPlatesAdvanced(1)
    
    for i in range(5):
        dp.push(i)
        print(f"  push({i})")
    
    for i in range(6):  # Try to pop more than pushed
        result = dp.pop()
        print(f"  pop(): {result}")
    
    # Test popAtStack on empty/invalid indices
    print(f"\nInvalid popAtStack:")
    dp = DinnerPlatesOptimal(3)
    
    results = [
        dp.popAtStack(0),   # Empty stack
        dp.popAtStack(10),  # Non-existent stack
        dp.popAtStack(-1)   # Negative index
    ]
    
    for i, result in enumerate(results):
        print(f"  popAtStack test {i+1}: {result}")
    
    # Test large capacity
    print(f"\nLarge capacity:")
    dp = DinnerPlatesSimple(1000)
    
    # Push many elements
    for i in range(2500):  # Should create 3 stacks (1000 + 1000 + 500)
        dp.push(i)
    
    print(f"  Pushed 2500 elements with capacity 1000")
    
    # Pop from middle stack
    middle_result = dp.popAtStack(1)
    print(f"  popAtStack(1): {middle_result}")
    
    # Push one more (should go to stack 1 now)
    dp.push(9999)
    print(f"  push(9999) after popAtStack")
    
    # Pop from stack 1 again
    result = dp.popAtStack(1)
    print(f"  popAtStack(1): {result}")

def test_performance_comparison():
    """Test performance of different implementations"""
    print("\n=== Testing Performance Comparison ===")
    
    import time
    
    implementations = [
        ("Simple", DinnerPlatesSimple),
        ("Optimal", DinnerPlatesOptimal),
        ("Advanced", DinnerPlatesAdvanced)
    ]
    
    capacity = 10
    num_operations = 5000
    
    for name, DinnerPlatesClass in implementations:
        dp = DinnerPlatesClass(capacity)
        
        # Time push operations
        start_time = time.time()
        for i in range(num_operations):
            dp.push(i)
        push_time = (time.time() - start_time) * 1000
        
        # Time pop operations
        start_time = time.time()
        for _ in range(num_operations // 2):
            dp.pop()
        pop_time = (time.time() - start_time) * 1000
        
        # Time popAtStack operations
        start_time = time.time()
        import random
        for _ in range(num_operations // 4):
            index = random.randint(0, num_operations // capacity)
            dp.popAtStack(index)
        pop_at_stack_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Push: {push_time:.2f}ms")
        print(f"    Pop: {pop_time:.2f}ms")
        print(f"    PopAtStack: {pop_at_stack_time:.2f}ms")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    dp = DinnerPlatesAdvanced(3)
    
    # Create complex state
    operations = [
        # Fill first few stacks
        *[("push", i) for i in range(10)],
        # Pop from middle stacks
        ("popAtStack", 1), ("popAtStack", 1), 
        ("popAtStack", 0),
        # Push more
        *[("push", i + 100) for i in range(5)],
        # Pop from right
        ("pop"), ("pop"),
    ]
    
    print("Executing complex operations:")
    for op, *args in operations:
        if op == "push":
            dp.push(args[0])
        elif op == "pop":
            result = dp.pop()
        elif op == "popAtStack":
            result = dp.popAtStack(args[0])
    
    # Get statistics
    stats = dp.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Get state
    state = dp.get_state()
    print(f"\nCurrent state:")
    for key, value in state.items():
        print(f"  {key}: {value}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Web server request handling
    print("Application 1: Web Server Request Pool")
    
    request_pool = DinnerPlatesAdvanced(100)  # Each pool handles 100 requests
    
    # Simulate incoming requests
    print("  Simulating incoming requests:")
    
    for minute in range(5):
        requests_this_minute = (minute + 1) * 50  # Increasing load
        
        print(f"    Minute {minute + 1}: {requests_this_minute} requests")
        
        # Add requests to pools
        for req_id in range(requests_this_minute):
            request_id = minute * 1000 + req_id
            request_pool.push(request_id)
        
        # Process some requests from specific pools
        if minute >= 2:
            # Process from first pool (priority processing)
            processed = request_pool.popAtStack(0)
            if processed != -1:
                print(f"      Priority processed request: {processed}")
    
    # Show final state
    stats = request_pool.get_statistics()
    print(f"  Final request pool stats:")
    print(f"    Total pools used: {stats['max_stacks_used']}")
    print(f"    Pending requests: {stats['total_elements']}")
    
    # Application 2: Memory management system
    print(f"\nApplication 2: Memory Block Management")
    
    memory_manager = DinnerPlatesMemoryOptimized(64)  # 64 blocks per chunk
    
    # Simulate memory allocation/deallocation
    print("  Simulating memory operations:")
    
    # Allocate memory blocks
    allocated_blocks = []
    for i in range(200):
        memory_manager.push(i)  # Block ID
        allocated_blocks.append(i)
    
    print(f"    Allocated 200 memory blocks")
    
    # Deallocate from specific chunks (fragmentation)
    for chunk_id in [0, 2, 4]:
        freed_block = memory_manager.popAtStack(chunk_id)
        if freed_block != -1:
            print(f"    Freed block {freed_block} from chunk {chunk_id}")
    
    # Allocate more (should reuse fragmented chunks)
    for i in range(10):
        memory_manager.push(1000 + i)
    
    print(f"    Allocated 10 more blocks (should reuse fragmented chunks)")
    
    # Show memory statistics
    memory_stats = memory_manager.get_memory_stats()
    print(f"  Memory system stats:")
    for key, value in memory_stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.3f}")
        else:
            print(f"    {key}: {value}")
    
    # Application 3: Task scheduling system
    print(f"\nApplication 3: Multi-Priority Task Scheduler")
    
    task_scheduler = DinnerPlatesAdvanced(50)  # 50 tasks per priority level
    
    # Simulate task submission with different priorities
    print("  Task scheduling simulation:")
    
    task_types = [
        ("High Priority", 20),
        ("Medium Priority", 100),
        ("Low Priority", 150),
        ("Background", 80)
    ]
    
    for task_type, count in task_types:
        print(f"    Submitting {count} {task_type} tasks")
        for i in range(count):
            task_id = hash(f"{task_type}_{i}") % 10000
            task_scheduler.push(task_id)
    
    # Process high priority tasks first (from first few stacks)
    print(f"  Processing high priority tasks:")
    for priority_stack in range(2):  # Process from first 2 stacks (high priority)
        processed_count = 0
        while processed_count < 10:  # Process 10 tasks from each
            task = task_scheduler.popAtStack(priority_stack)
            if task == -1:
                break
            processed_count += 1
        
        print(f"    Processed {processed_count} tasks from priority level {priority_stack}")
    
    # Process remaining tasks in LIFO order
    background_processed = 0
    while background_processed < 20:
        task = task_scheduler.pop()
        if task == -1:
            break
        background_processed += 1
    
    print(f"    Processed {background_processed} background tasks")
    
    # Show final scheduler state
    final_stats = task_scheduler.get_statistics()
    print(f"  Scheduler final state:")
    print(f"    Remaining tasks: {final_stats['total_elements']}")
    print(f"    Active priority levels: {final_stats['current_active_stacks']}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Standard", DinnerPlatesOptimal),
        ("Memory Optimized", DinnerPlatesMemoryOptimized)
    ]
    
    capacity = 100
    
    for name, DinnerPlatesClass in implementations:
        dp = DinnerPlatesClass(capacity)
        
        print(f"\n{name}:")
        
        # Phase 1: Fill many stacks
        for i in range(5000):
            dp.push(i)
        
        print(f"  After pushing 5000 elements:")
        if hasattr(dp, 'get_memory_stats'):
            memory_stats = dp.get_memory_stats()
            for key, value in memory_stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
        
        # Phase 2: Create fragmentation
        for stack_id in range(0, 25, 2):  # Pop from every other stack
            for _ in range(50):  # Pop 50 elements from each
                dp.popAtStack(stack_id)
        
        print(f"  After creating fragmentation:")
        if hasattr(dp, 'get_memory_stats'):
            memory_stats = dp.get_memory_stats()
            for key, value in memory_stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
        
        # Phase 3: Add more elements (should reuse fragmented space)
        for i in range(1000):
            dp.push(10000 + i)
        
        print(f"  After refilling fragmented space:")
        if hasattr(dp, 'get_memory_stats'):
            memory_stats = dp.get_memory_stats()
            for key, value in memory_stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")

def stress_test_dinner_plates():
    """Stress test dinner plates"""
    print("\n=== Stress Testing Dinner Plates ===")
    
    import time
    import random
    
    dp = DinnerPlatesOptimal(1000)  # Large capacity
    
    # Stress test parameters
    num_operations = 50000
    
    print(f"Stress test: {num_operations} mixed operations")
    
    start_time = time.time()
    
    # Mixed operations
    push_count = 0
    pop_count = 0
    pop_at_stack_count = 0
    
    for i in range(num_operations):
        operation = random.choices(
            ['push', 'pop', 'popAtStack'], 
            weights=[0.6, 0.2, 0.2]  # 60% push, 20% pop, 20% popAtStack
        )[0]
        
        if operation == 'push':
            dp.push(i)
            push_count += 1
        
        elif operation == 'pop':
            dp.pop()
            pop_count += 1
        
        elif operation == 'popAtStack':
            # Pop from random stack (might be empty)
            stack_index = random.randint(0, 100)
            dp.popAtStack(stack_index)
            pop_at_stack_count += 1
    
    elapsed = (time.time() - start_time) * 1000
    
    print(f"  Completed in {elapsed:.2f}ms")
    print(f"  Operations: {push_count} pushes, {pop_count} pops, {pop_at_stack_count} popAtStack")
    print(f"  Average: {elapsed/num_operations:.4f}ms per operation")

def benchmark_stack_patterns():
    """Benchmark different stack usage patterns"""
    print("\n=== Benchmarking Stack Patterns ===")
    
    import time
    
    patterns = [
        ("Sequential fill", "sequential"),
        ("Random access", "random"),
        ("Fragmented access", "fragmented"),
        ("LIFO heavy", "lifo")
    ]
    
    capacity = 100
    operations = 10000
    
    for pattern_name, pattern_type in patterns:
        dp = DinnerPlatesAdvanced(capacity)
        
        start_time = time.time()
        
        if pattern_type == "sequential":
            # Fill sequentially, pop sequentially
            for i in range(operations):
                dp.push(i)
            for _ in range(operations // 2):
                dp.pop()
        
        elif pattern_type == "random":
            # Random mix of operations
            import random
            for i in range(operations):
                if random.random() < 0.7:  # 70% push
                    dp.push(i)
                else:
                    if random.random() < 0.5:
                        dp.pop()
                    else:
                        dp.popAtStack(random.randint(0, 50))
        
        elif pattern_type == "fragmented":
            # Create fragmentation pattern
            # Fill stacks then pop from middle
            for i in range(operations // 2):
                dp.push(i)
            
            # Create fragmentation
            for stack_idx in range(0, 20, 2):
                for _ in range(10):
                    dp.popAtStack(stack_idx)
            
            # Refill
            for i in range(operations // 2):
                dp.push(operations + i)
        
        elif pattern_type == "lifo":
            # Heavy LIFO usage
            for i in range(operations):
                if i % 3 == 0:
                    dp.pop()
                else:
                    dp.push(i)
        
        elapsed = (time.time() - start_time) * 1000
        
        # Get final statistics
        if hasattr(dp, 'get_statistics'):
            stats = dp.get_statistics()
            print(f"  {pattern_name}: {elapsed:.2f}ms")
            print(f"    Final elements: {stats.get('total_elements', 'N/A')}")
            print(f"    Max stacks used: {stats.get('max_stacks_used', 'N/A')}")
        else:
            print(f"  {pattern_name}: {elapsed:.2f}ms")

def test_concurrent_access():
    """Test concurrent access patterns"""
    print("\n=== Testing Concurrent Access ===")
    
    import threading
    import time
    import random
    
    dp = DinnerPlatesConcurrent(50)
    
    # Test concurrent operations
    num_threads = 4
    operations_per_thread = 1000
    
    def worker_thread(thread_id: int, results: list):
        """Worker thread for concurrent operations"""
        start_time = time.time()
        
        for i in range(operations_per_thread):
            operation = random.choice(['push', 'pop', 'popAtStack'])
            
            if operation == 'push':
                dp.push(thread_id * 10000 + i)
            elif operation == 'pop':
                dp.pop()
            elif operation == 'popAtStack':
                dp.popAtStack(random.randint(0, 20))
        
        elapsed = time.time() - start_time
        results.append(elapsed)
    
    print(f"Running {num_threads} concurrent threads...")
    
    # Start threads
    threads = []
    results = []
    
    overall_start = time.time()
    
    for thread_id in range(num_threads):
        thread = threading.Thread(target=worker_thread, args=(thread_id, results))
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    overall_time = time.time() - overall_start
    total_operations = num_threads * operations_per_thread
    
    print(f"  Total time: {overall_time:.2f}s")
    print(f"  Total operations: {total_operations}")
    print(f"  Throughput: {total_operations/overall_time:.0f} ops/sec")
    print(f"  Thread times: {[f'{t:.2f}s' for t in results]}")
    
    # Get final statistics
    final_stats = dp.get_stats()
    print(f"  Final stats: {final_stats}")

if __name__ == "__main__":
    test_dinner_plates_basic()
    test_dinner_plates_edge_cases()
    test_performance_comparison()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    stress_test_dinner_plates()
    benchmark_stack_patterns()
    test_concurrent_access()

"""
Dinner Plate Stacks Design demonstrates key concepts:

Core Approaches:
1. Simple - Linear search for available stacks
2. Optimal - Min-heap for efficient leftmost available tracking
3. Advanced - Multiple data structures with caching and statistics
4. Memory Optimized - Sparse representation with cleanup
5. Concurrent - Thread-safe implementation with locking

Key Design Principles:
- Efficient leftmost available stack tracking
- Rightmost non-empty stack caching for pop operations
- Memory management for large-scale systems
- Balancing space and time complexity

Performance Characteristics:
- Simple: O(n) push/pop, O(1) popAtStack
- Optimal: O(log n) push, O(1) pop amortized, O(log n) popAtStack
- Advanced: Enhanced with caching and statistics overhead
- Memory Optimized: Sparse storage saves memory

Real-world Applications:
- Web server request pool management
- Memory block allocation systems
- Multi-priority task scheduling
- Load balancing across server instances
- Database connection pool management
- Resource allocation in distributed systems

The optimal heap-based approach provides the best balance
of performance for most use cases, offering logarithmic
complexity for push operations while maintaining efficient
pop and popAtStack operations.
"""
