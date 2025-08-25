"""
895. Maximum Frequency Stack - Multiple Approaches
Difficulty: Hard

Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the FreqStack class:
- FreqStack() constructs an empty frequency stack.
- void push(int val) pushes an integer val onto the top of the stack.
- int pop() removes and returns the most frequent element in the stack. If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.
"""

from typing import Dict, List
from collections import defaultdict, Counter
import heapq

class FreqStackOptimal:
    """
    Approach 1: Frequency Groups with Stacks
    
    Maintain separate stacks for each frequency level.
    
    Time Complexity: O(1) for both push and pop
    Space Complexity: O(n) where n is number of elements
    """
    
    def __init__(self):
        self.freq = defaultdict(int)  # Element -> frequency
        self.group = defaultdict(list)  # Frequency -> stack of elements
        self.max_freq = 0
    
    def push(self, val: int) -> None:
        # Increment frequency
        self.freq[val] += 1
        f = self.freq[val]
        
        # Update max frequency
        self.max_freq = max(self.max_freq, f)
        
        # Add to frequency group
        self.group[f].append(val)
    
    def pop(self) -> int:
        # Get element from highest frequency group
        val = self.group[self.max_freq].pop()
        
        # Decrement frequency
        self.freq[val] -= 1
        
        # Update max frequency if needed
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        
        return val

class FreqStackHeap:
    """
    Approach 2: Max Heap with Ordering
    
    Use max heap with frequency and insertion order.
    
    Time Complexity: O(log n) for both push and pop
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.freq = defaultdict(int)
        self.heap = []
        self.order = 0  # To handle ties by recency
    
    def push(self, val: int) -> None:
        self.freq[val] += 1
        self.order += 1
        
        # Push (-frequency, -order, val) for max heap behavior
        heapq.heappush(self.heap, (-self.freq[val], -self.order, val))
    
    def pop(self) -> int:
        # Find the most frequent element
        while self.heap:
            neg_freq, neg_order, val = heapq.heappop(self.heap)
            freq = -neg_freq
            
            # Check if this is still the current frequency
            if freq == self.freq[val]:
                self.freq[val] -= 1
                if self.freq[val] == 0:
                    del self.freq[val]
                return val
        
        return -1  # Should never reach here

class FreqStackSimple:
    """
    Approach 3: Simple with Linear Search
    
    Store elements with metadata and search linearly.
    
    Time Complexity: O(1) for push, O(n) for pop
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.stack = []  # List of (val, freq, order)
        self.freq = defaultdict(int)
        self.order = 0
    
    def push(self, val: int) -> None:
        self.freq[val] += 1
        self.order += 1
        self.stack.append((val, self.freq[val], self.order))
    
    def pop(self) -> int:
        if not self.stack:
            return -1
        
        # Find element with max frequency (and latest order for ties)
        max_freq = 0
        max_order = -1
        pop_index = -1
        
        for i in range(len(self.stack) - 1, -1, -1):  # Reverse order for recency
            val, freq, order = self.stack[i]
            
            if freq > max_freq or (freq == max_freq and order > max_order):
                max_freq = freq
                max_order = order
                pop_index = i
        
        if pop_index != -1:
            val, _, _ = self.stack.pop(pop_index)
            self.freq[val] -= 1
            if self.freq[val] == 0:
                del self.freq[val]
            return val
        
        return -1

class FreqStackAdvanced:
    """
    Approach 4: Advanced with Statistics and Analysis
    
    Enhanced version with operation tracking and analytics.
    
    Time Complexity: O(1) for push and pop
    Space Complexity: O(n + k) where k is distinct frequencies
    """
    
    def __init__(self):
        self.freq = defaultdict(int)
        self.group = defaultdict(list)
        self.max_freq = 0
        
        # Statistics
        self.push_count = 0
        self.pop_count = 0
        self.unique_elements = set()
        self.frequency_distribution = defaultdict(int)
        
    def push(self, val: int) -> None:
        # Update statistics
        self.push_count += 1
        self.unique_elements.add(val)
        
        # Remove old frequency count
        if val in self.freq:
            old_freq = self.freq[val]
            self.frequency_distribution[old_freq] -= 1
            if self.frequency_distribution[old_freq] == 0:
                del self.frequency_distribution[old_freq]
        
        # Update frequency
        self.freq[val] += 1
        f = self.freq[val]
        
        # Update frequency distribution
        self.frequency_distribution[f] += 1
        
        # Update max frequency
        self.max_freq = max(self.max_freq, f)
        
        # Add to frequency group
        self.group[f].append(val)
    
    def pop(self) -> int:
        if self.max_freq == 0:
            return -1
        
        # Update statistics
        self.pop_count += 1
        
        # Get element from highest frequency group
        val = self.group[self.max_freq].pop()
        
        # Update frequency distribution
        self.frequency_distribution[self.max_freq] -= 1
        if self.frequency_distribution[self.max_freq] == 0:
            del self.frequency_distribution[self.max_freq]
        
        # Decrement frequency
        old_freq = self.freq[val]
        self.freq[val] -= 1
        
        if self.freq[val] == 0:
            del self.freq[val]
            self.unique_elements.discard(val)
        else:
            # Update frequency distribution for new frequency
            new_freq = self.freq[val]
            self.frequency_distribution[new_freq] += 1
        
        # Update max frequency if needed
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        
        return val
    
    def get_statistics(self) -> dict:
        """Get operation statistics"""
        return {
            'push_count': self.push_count,
            'pop_count': self.pop_count,
            'current_size': self.push_count - self.pop_count,
            'unique_elements_count': len(self.unique_elements),
            'max_frequency': self.max_freq,
            'frequency_levels': len(self.group),
            'frequency_distribution': dict(self.frequency_distribution)
        }
    
    def get_state(self) -> dict:
        """Get current state for debugging"""
        return {
            'frequencies': dict(self.freq),
            'groups': {f: list(stack) for f, stack in self.group.items()},
            'max_freq': self.max_freq
        }

class FreqStackMemoryOptimized:
    """
    Approach 5: Memory-Optimized Version
    
    Optimized for memory usage with lazy cleanup.
    
    Time Complexity: O(1) amortized for push and pop
    Space Complexity: O(n) with better memory management
    """
    
    def __init__(self):
        self.freq = {}
        self.group = {}
        self.max_freq = 0
        self.cleanup_threshold = 1000
        self.operations_since_cleanup = 0
    
    def push(self, val: int) -> None:
        # Increment frequency
        self.freq[val] = self.freq.get(val, 0) + 1
        f = self.freq[val]
        
        # Update max frequency
        if f > self.max_freq:
            self.max_freq = f
        
        # Add to frequency group
        if f not in self.group:
            self.group[f] = []
        self.group[f].append(val)
        
        self.operations_since_cleanup += 1
        
        # Periodic cleanup
        if self.operations_since_cleanup >= self.cleanup_threshold:
            self._cleanup()
    
    def pop(self) -> int:
        if self.max_freq == 0:
            return -1
        
        # Get element from highest frequency group
        val = self.group[self.max_freq].pop()
        
        # Decrement frequency
        self.freq[val] -= 1
        if self.freq[val] == 0:
            del self.freq[val]
        
        # Update max frequency if needed
        if not self.group[self.max_freq]:
            self._find_new_max_freq()
        
        self.operations_since_cleanup += 1
        
        return val
    
    def _find_new_max_freq(self) -> None:
        """Find new max frequency after current max becomes empty"""
        while self.max_freq > 0 and not self.group.get(self.max_freq):
            if self.max_freq in self.group:
                del self.group[self.max_freq]
            self.max_freq -= 1
    
    def _cleanup(self) -> None:
        """Clean up empty frequency groups"""
        empty_freqs = [f for f, stack in self.group.items() if not stack]
        for f in empty_freqs:
            del self.group[f]
        
        self.operations_since_cleanup = 0


def test_freq_stack_basic():
    """Test basic FreqStack functionality"""
    print("=== Testing Basic FreqStack Functionality ===")
    
    implementations = [
        ("Optimal Groups", FreqStackOptimal),
        ("Max Heap", FreqStackHeap),
        ("Simple Linear", FreqStackSimple),
        ("Advanced", FreqStackAdvanced),
        ("Memory Optimized", FreqStackMemoryOptimized)
    ]
    
    for name, FreqStackClass in implementations:
        print(f"\n{name}:")
        
        fs = FreqStackClass()
        
        # Test sequence from problem description
        operations = [
            ("push", 5), ("push", 7), ("push", 5), ("push", 7),
            ("push", 4), ("push", 5), ("pop", None), ("pop", None),
            ("pop", None), ("pop", None)
        ]
        
        for op, val in operations:
            if op == "push":
                fs.push(val)
                print(f"  push({val})")
            else:  # pop
                result = fs.pop()
                print(f"  pop(): {result}")

def test_freq_stack_edge_cases():
    """Test FreqStack edge cases"""
    print("\n=== Testing FreqStack Edge Cases ===")
    
    fs = FreqStackOptimal()
    
    # Test pop from empty stack
    print("Empty stack:")
    result = fs.pop()
    print(f"  pop() from empty: {result}")
    
    # Test single element
    print(f"\nSingle element:")
    fs.push(42)
    print(f"  push(42)")
    result = fs.pop()
    print(f"  pop(): {result}")
    result = fs.pop()
    print(f"  pop() again: {result}")
    
    # Test all same frequency
    print(f"\nAll elements same frequency:")
    elements = [1, 2, 3, 4, 5]
    
    for elem in elements:
        fs.push(elem)
        print(f"  push({elem})")
    
    print(f"  Popping (should be LIFO order):")
    while True:
        result = fs.pop()
        if result == -1:
            break
        print(f"    pop(): {result}")

def test_frequency_behavior():
    """Test frequency-based behavior"""
    print("\n=== Testing Frequency-Based Behavior ===")
    
    fs = FreqStackAdvanced()
    
    # Create specific frequency pattern
    # Element frequencies: 1->3 times, 2->2 times, 3->1 time
    pushes = [1, 2, 1, 3, 2, 1]
    
    print("Building frequency pattern:")
    for val in pushes:
        fs.push(val)
        print(f"  push({val})")
    
    # Get state
    state = fs.get_state()
    print(f"\nCurrent state:")
    print(f"  Frequencies: {state['frequencies']}")
    print(f"  Groups: {state['groups']}")
    
    # Pop and verify order
    print(f"\nPopping (should respect frequency then recency):")
    expected_order = [1, 1, 2, 1, 2, 3]  # Based on frequency and recency
    
    for i in range(len(pushes)):
        result = fs.pop()
        print(f"  pop(): {result}")

def test_performance():
    """Test FreqStack performance"""
    print("\n=== Testing FreqStack Performance ===")
    
    import time
    
    implementations = [
        ("Optimal Groups", FreqStackOptimal),
        ("Max Heap", FreqStackHeap),
        ("Simple Linear", FreqStackSimple)
    ]
    
    num_operations = 10000
    
    for name, FreqStackClass in implementations:
        fs = FreqStackClass()
        
        # Time push operations
        start_time = time.time()
        for i in range(num_operations):
            fs.push(i % 100)  # Limited value range for frequency buildup
        push_time = (time.time() - start_time) * 1000
        
        # Time pop operations
        start_time = time.time()
        for _ in range(min(5000, num_operations)):  # Pop half
            fs.pop()
        pop_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    {num_operations} pushes: {push_time:.2f}ms")
        print(f"    5000 pops: {pop_time:.2f}ms")

def test_advanced_features():
    """Test advanced features"""
    print("\n=== Testing Advanced Features ===")
    
    fs = FreqStackAdvanced()
    
    # Perform various operations
    operations = [
        (1, 3), (2, 2), (3, 4), (1, 2), (4, 1)  # (value, count)
    ]
    
    print("Building complex frequency distribution:")
    for val, count in operations:
        for _ in range(count):
            fs.push(val)
        print(f"  Added {val} {count} times")
    
    # Get statistics
    stats = fs.get_statistics()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Pop a few and check updated stats
    print(f"\nAfter popping 3 elements:")
    for _ in range(3):
        popped = fs.pop()
        print(f"  Popped: {popped}")
    
    updated_stats = fs.get_statistics()
    print(f"  Updated current size: {updated_stats['current_size']}")
    print(f"  Updated max frequency: {updated_stats['max_frequency']}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: LFU Cache with recency
    print("Application 1: LFU Cache with Recency Tie-Breaking")
    
    cache_freq = FreqStackOptimal()
    
    # Simulate cache access pattern
    accesses = ["page1", "page2", "page1", "page3", "page2", "page1", "page4"]
    page_to_id = {"page1": 1, "page2": 2, "page3": 3, "page4": 4}
    
    print("  Cache access pattern:")
    for page in accesses:
        page_id = page_to_id[page]
        cache_freq.push(page_id)
        print(f"    Access {page} (id: {page_id})")
    
    # Simulate cache eviction (pop most frequent)
    print(f"  Cache eviction order:")
    for _ in range(3):
        evicted_id = cache_freq.pop()
        evicted_page = next(page for page, pid in page_to_id.items() if pid == evicted_id)
        print(f"    Evicted: {evicted_page} (id: {evicted_id})")
    
    # Application 2: Priority task scheduling
    print(f"\nApplication 2: Priority Task Scheduling")
    
    task_scheduler = FreqStackOptimal()
    
    # Tasks with different priorities (frequency = priority level)
    tasks = [
        ("urgent_task", 3), ("normal_task", 1), ("urgent_task", 1),
        ("low_priority", 1), ("normal_task", 1), ("urgent_task", 1)
    ]
    
    task_to_id = {"urgent_task": 1, "normal_task": 2, "low_priority": 3}
    
    print("  Task submission:")
    for task, priority in tasks:
        task_id = task_to_id[task]
        for _ in range(priority):
            task_scheduler.push(task_id)
        print(f"    Submit {task} with priority {priority}")
    
    print(f"  Task execution order:")
    for _ in range(5):
        task_id = task_scheduler.pop()
        task_name = next(name for name, tid in task_to_id.items() if tid == task_id)
        print(f"    Execute: {task_name}")
    
    # Application 3: Undo system with frequency
    print(f"\nApplication 3: Smart Undo System")
    
    undo_system = FreqStackOptimal()
    
    # Actions with different frequencies (common actions pushed more)
    actions = [
        ("type", 10), ("delete", 3), ("copy", 2), ("paste", 2), 
        ("type", 5), ("undo", 1), ("save", 1)
    ]
    
    action_to_id = {"type": 1, "delete": 2, "copy": 3, "paste": 4, "undo": 5, "save": 6}
    
    print("  Action frequency recording:")
    for action, freq in actions:
        action_id = action_to_id[action]
        for _ in range(freq):
            undo_system.push(action_id)
        print(f"    {action}: {freq} times")
    
    print(f"  Smart undo suggestions (most frequent first):")
    for _ in range(4):
        action_id = undo_system.pop()
        action_name = next(name for name, aid in action_to_id.items() if aid == action_id)
        print(f"    Suggest undo: {action_name}")

def test_memory_efficiency():
    """Test memory efficiency"""
    print("\n=== Testing Memory Efficiency ===")
    
    implementations = [
        ("Optimal Groups", FreqStackOptimal),
        ("Memory Optimized", FreqStackMemoryOptimized)
    ]
    
    for name, FreqStackClass in implementations:
        fs = FreqStackClass()
        
        # Add many elements with varying frequencies
        num_elements = 1000
        for i in range(num_elements):
            val = i % 100  # 100 unique values
            fs.push(val)
        
        # Estimate memory usage
        if hasattr(fs, 'freq'):
            freq_memory = len(fs.freq)
        else:
            freq_memory = 100
        
        if hasattr(fs, 'group'):
            group_memory = sum(len(stack) for stack in fs.group.values())
        else:
            group_memory = num_elements
        
        total_memory = freq_memory + group_memory
        
        print(f"  {name}:")
        print(f"    Frequency map: {freq_memory} entries")
        print(f"    Group stacks: {group_memory} total elements")
        print(f"    Total memory: ~{total_memory} units")

def test_stress_scenarios():
    """Test stress scenarios"""
    print("\n=== Testing Stress Scenarios ===")
    
    # Stress test 1: High frequency single element
    print("Stress test 1: High frequency single element")
    
    fs = FreqStackOptimal()
    
    # Push same element many times
    high_freq_count = 1000
    for _ in range(high_freq_count):
        fs.push(42)
    
    # Add a few other elements
    for i in range(5):
        fs.push(i)
    
    print(f"  Added element 42 {high_freq_count} times")
    print(f"  Added elements 0-4 once each")
    
    # Pop a few times
    for i in range(10):
        result = fs.pop()
        print(f"    Pop {i+1}: {result}")
    
    # Stress test 2: Many unique elements
    print(f"\nStress test 2: Many unique elements with equal frequency")
    
    fs2 = FreqStackMemoryOptimized()
    
    # Add many unique elements
    unique_count = 5000
    for i in range(unique_count):
        fs2.push(i)
    
    print(f"  Added {unique_count} unique elements")
    
    # Pop some to test performance
    import time
    start_time = time.time()
    
    pop_count = 1000
    for _ in range(pop_count):
        fs2.pop()
    
    elapsed = (time.time() - start_time) * 1000
    print(f"  Popped {pop_count} elements in {elapsed:.2f}ms")

def test_frequency_distribution():
    """Test various frequency distributions"""
    print("\n=== Testing Frequency Distributions ===")
    
    fs = FreqStackAdvanced()
    
    # Create Zipfian-like distribution
    values = [1] * 50 + [2] * 25 + [3] * 12 + [4] * 6 + [5] * 3 + [6] * 1
    
    print("Creating Zipfian-like frequency distribution:")
    print(f"  Element frequencies: 1->50, 2->25, 3->12, 4->6, 5->3, 6->1")
    
    for val in values:
        fs.push(val)
    
    # Get statistics
    stats = fs.get_statistics()
    print(f"  Total elements: {stats['current_size']}")
    print(f"  Max frequency: {stats['max_frequency']}")
    print(f"  Frequency distribution: {stats['frequency_distribution']}")
    
    # Pop elements and track order
    print(f"\nPopping elements (first 10):")
    pop_order = []
    
    for i in range(10):
        popped = fs.pop()
        pop_order.append(popped)
        print(f"    Pop {i+1}: {popped}")
    
    # Analyze pop order
    from collections import Counter
    pop_freq = Counter(pop_order)
    print(f"  Pop frequency distribution: {dict(pop_freq)}")

def benchmark_operation_ratios():
    """Benchmark different push/pop ratios"""
    print("\n=== Benchmarking Operation Ratios ===")
    
    import time
    
    fs = FreqStackOptimal()
    
    ratios = [
        ("Push heavy (10:1)", 10, 1),
        ("Balanced (1:1)", 1, 1),
        ("Pop heavy (1:3)", 1, 3)
    ]
    
    for ratio_name, push_ratio, pop_ratio in ratios:
        print(f"\n{ratio_name}:")
        
        # Reset stack
        fs = FreqStackOptimal()
        
        # Pre-populate for pop-heavy test
        if pop_ratio > push_ratio:
            for i in range(1000):
                fs.push(i % 50)
        
        start_time = time.time()
        
        cycle_ops = 0
        total_ops = 1000
        
        for _ in range(total_ops // (push_ratio + pop_ratio)):
            # Push operations
            for _ in range(push_ratio):
                fs.push(cycle_ops % 100)
                cycle_ops += 1
            
            # Pop operations
            for _ in range(pop_ratio):
                fs.pop()
                cycle_ops += 1
        
        elapsed = (time.time() - start_time) * 1000
        
        print(f"    {cycle_ops} operations in {elapsed:.2f}ms")
        print(f"    Average: {elapsed/cycle_ops:.4f}ms per operation")

if __name__ == "__main__":
    test_freq_stack_basic()
    test_freq_stack_edge_cases()
    test_frequency_behavior()
    test_performance()
    test_advanced_features()
    demonstrate_applications()
    test_memory_efficiency()
    test_stress_scenarios()
    test_frequency_distribution()
    benchmark_operation_ratios()

"""
Maximum Frequency Stack Design demonstrates key concepts:

Core Approaches:
1. Optimal Groups - Separate stacks per frequency level O(1)
2. Max Heap - Priority queue with frequency and order O(log n)
3. Simple Linear - Linear search for max frequency O(n) pop
4. Advanced - Enhanced with statistics and analytics
5. Memory Optimized - Lazy cleanup and memory management

Key Design Principles:
- Frequency-based prioritization with recency tie-breaking
- Separate data structures for different frequency levels
- Efficient max frequency tracking and updates
- Memory management for long-running systems

Data Structure Components:
- freq: HashMap element -> frequency count
- group: HashMap frequency -> stack of elements
- max_freq: Current maximum frequency level
- Recency handled by stack order within frequency groups

Performance Characteristics:
- Optimal: O(1) push, O(1) pop, O(n) space
- Heap: O(log n) push, O(log n) pop, O(n) space  
- Simple: O(1) push, O(n) pop, O(n) space
- All maintain frequency ordering with recency tie-breaking

Real-world Applications:
- LFU cache with recency tie-breaking
- Priority task scheduling with frequency weighting
- Smart undo systems prioritizing common actions
- Hot data identification in databases
- Frequency-based recommendation systems
- Load balancer request prioritization

The frequency groups approach is optimal for this problem,
providing O(1) operations while correctly handling both
frequency prioritization and recency-based tie-breaking.
"""
