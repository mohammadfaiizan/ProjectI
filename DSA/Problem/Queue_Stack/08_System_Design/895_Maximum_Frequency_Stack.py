"""
895. Maximum Frequency Stack - Multiple Approaches
Difficulty: Hard

Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the FreqStack class:
- FreqStack() constructs an empty frequency stack.
- void push(int val) pushes an integer val onto the top of the stack.
- int pop() removes and returns the most frequent element in the stack.

If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.
"""

from typing import Dict, List
from collections import defaultdict, Counter
import heapq

class FreqStackOptimal:
    """
    Approach 1: Frequency Groups with Stacks (Optimal)
    
    Use frequency groups where each frequency has its own stack.
    
    Time: O(1) for both push and pop, Space: O(n)
    """
    
    def __init__(self):
        self.freq = defaultdict(int)  # val -> frequency
        self.group = defaultdict(list)  # frequency -> stack of values
        self.max_freq = 0
    
    def push(self, val: int) -> None:
        """Push value to frequency stack"""
        # Increment frequency
        self.freq[val] += 1
        f = self.freq[val]
        
        # Update max frequency
        if f > self.max_freq:
            self.max_freq = f
        
        # Add to frequency group
        self.group[f].append(val)
    
    def pop(self) -> int:
        """Pop most frequent element"""
        # Get value from highest frequency group
        val = self.group[self.max_freq].pop()
        
        # Decrement frequency
        self.freq[val] -= 1
        
        # Update max frequency if needed
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        
        return val


class FreqStackHeap:
    """
    Approach 2: Max Heap Implementation
    
    Use max heap to track elements by frequency and insertion order.
    
    Time: O(log n) for both operations, Space: O(n)
    """
    
    def __init__(self):
        self.freq = Counter()
        self.heap = []  # (-frequency, -insertion_order, val)
        self.insertion_order = 0
    
    def push(self, val: int) -> None:
        """Push value to frequency stack"""
        self.freq[val] += 1
        self.insertion_order += 1
        
        # Push to heap with negative values for max heap behavior
        heapq.heappush(self.heap, (-self.freq[val], -self.insertion_order, val))
    
    def pop(self) -> int:
        """Pop most frequent element"""
        # Find the most frequent element that hasn't been removed
        while self.heap:
            neg_freq, neg_order, val = heapq.heappop(self.heap)
            freq = -neg_freq
            
            # Check if this element still has the expected frequency
            if self.freq[val] == freq:
                self.freq[val] -= 1
                if self.freq[val] == 0:
                    del self.freq[val]
                return val
        
        return -1  # Should never reach here


class FreqStackList:
    """
    Approach 3: List-based Implementation
    
    Use list to maintain elements and search for most frequent.
    
    Time: O(n) for pop, O(1) for push, Space: O(n)
    """
    
    def __init__(self):
        self.stack = []  # (val, insertion_order)
        self.insertion_order = 0
    
    def push(self, val: int) -> None:
        """Push value to frequency stack"""
        self.insertion_order += 1
        self.stack.append((val, self.insertion_order))
    
    def pop(self) -> int:
        """Pop most frequent element"""
        if not self.stack:
            return -1
        
        # Count frequencies
        freq_count = Counter()
        for val, _ in self.stack:
            freq_count[val] += 1
        
        # Find max frequency
        max_freq = max(freq_count.values())
        
        # Find the most recent element with max frequency
        for i in range(len(self.stack) - 1, -1, -1):
            val, _ = self.stack[i]
            if freq_count[val] == max_freq:
                return self.stack.pop(i)[0]
        
        return -1


class FreqStackMultipleStacks:
    """
    Approach 4: Multiple Stacks Implementation
    
    Maintain separate stacks for each unique frequency.
    
    Time: O(1) for both operations, Space: O(n)
    """
    
    def __init__(self):
        self.freq = {}  # val -> frequency
        self.stacks = {}  # frequency -> stack
        self.max_freq = 0
    
    def push(self, val: int) -> None:
        """Push value to frequency stack"""
        # Update frequency
        if val not in self.freq:
            self.freq[val] = 0
        self.freq[val] += 1
        
        f = self.freq[val]
        
        # Update max frequency
        self.max_freq = max(self.max_freq, f)
        
        # Add to appropriate frequency stack
        if f not in self.stacks:
            self.stacks[f] = []
        self.stacks[f].append(val)
    
    def pop(self) -> int:
        """Pop most frequent element"""
        # Get from highest frequency stack
        val = self.stacks[self.max_freq].pop()
        
        # Update frequency
        self.freq[val] -= 1
        if self.freq[val] == 0:
            del self.freq[val]
        
        # Update max frequency if needed
        if not self.stacks[self.max_freq]:
            self.max_freq -= 1
        
        return val


def test_freq_stack_implementations():
    """Test frequency stack implementations"""
    
    implementations = [
        ("Optimal", FreqStackOptimal),
        ("Heap", FreqStackHeap),
        ("List", FreqStackList),
        ("Multiple Stacks", FreqStackMultipleStacks),
    ]
    
    test_cases = [
        {
            "operations": ["push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"],
            "values": [5, 7, 5, 7, 4, 5, None, None, None, None],
            "expected": [None, None, None, None, None, None, 5, 7, 5, 4],
            "description": "Example 1"
        },
        {
            "operations": ["push", "push", "push", "pop", "pop", "pop"],
            "values": [1, 1, 1, None, None, None],
            "expected": [None, None, None, 1, 1, 1],
            "description": "Same element multiple times"
        },
        {
            "operations": ["push", "push", "push", "pop", "push", "pop"],
            "values": [1, 2, 3, None, 1, None],
            "expected": [None, None, None, 3, None, 1],
            "description": "Mixed operations"
        },
    ]
    
    print("=== Testing Frequency Stack Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                freq_stack = impl_class()
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "push":
                        freq_stack.push(test_case["values"][i])
                        results.append(None)
                    elif op == "pop":
                        result = freq_stack.pop()
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:25} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:25} | ERROR: {str(e)[:40]}")


def demonstrate_freq_stack_mechanism():
    """Demonstrate frequency stack mechanism step by step"""
    print("\n=== Frequency Stack Mechanism Step-by-Step Demo ===")
    
    freq_stack = FreqStackOptimal()
    
    operations = [
        ("push", 5),
        ("push", 7),
        ("push", 5),
        ("push", 7),
        ("push", 4),
        ("push", 5),
        ("pop", None),
        ("pop", None),
        ("pop", None),
        ("pop", None),
    ]
    
    print("Strategy: Group elements by frequency, pop from highest frequency group")
    
    def print_stack_state():
        """Helper to print current stack state"""
        print(f"  Frequencies: {dict(freq_stack.freq)}")
        print(f"  Max frequency: {freq_stack.max_freq}")
        groups = {}
        for freq, stack in freq_stack.group.items():
            if stack:
                groups[freq] = stack.copy()
        print(f"  Frequency groups: {groups}")
    
    print(f"\nInitial state:")
    print_stack_state()
    
    for i, (op, value) in enumerate(operations):
        print(f"\nStep {i+1}: {op}({value if value is not None else ''})")
        
        if op == "push":
            freq_stack.push(value)
            print(f"  Pushed {value}")
        elif op == "pop":
            result = freq_stack.pop()
            print(f"  Popped {result}")
        
        print_stack_state()


def visualize_frequency_groups():
    """Visualize frequency groups"""
    print("\n=== Frequency Groups Visualization ===")
    
    freq_stack = FreqStackOptimal()
    
    # Build up the stack
    elements = [5, 7, 5, 7, 4, 5]
    
    print("Building frequency stack:")
    for elem in elements:
        freq_stack.push(elem)
        
        print(f"\nAfter pushing {elem}:")
        print(f"  Element frequencies:")
        for val, freq in freq_stack.freq.items():
            print(f"    {val}: {freq} times")
        
        print(f"  Frequency groups:")
        for freq in sorted(freq_stack.group.keys()):
            if freq_stack.group[freq]:
                print(f"    Frequency {freq}: {freq_stack.group[freq]}")
    
    print(f"\nPopping elements:")
    while freq_stack.max_freq > 0:
        popped = freq_stack.pop()
        print(f"  Popped: {popped}")
        print(f"  Max frequency now: {freq_stack.max_freq}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Recently used files with frequency
    print("1. Recently Used Files with Frequency Priority:")
    file_stack = FreqStackOptimal()
    
    file_operations = [
        ("open", "document.txt"),
        ("open", "presentation.ppt"),
        ("open", "document.txt"),  # More frequent
        ("open", "spreadsheet.xls"),
        ("open", "document.txt"),  # Most frequent
        ("open", "presentation.ppt"),
        ("get_recent", None),  # Should return document.txt (most frequent)
        ("get_recent", None),  # Should return presentation.ppt
        ("get_recent", None),  # Should return document.txt again
    ]
    
    print("  File access simulation:")
    for action, filename in file_operations:
        if action == "open":
            # Use hash of filename as integer
            file_id = hash(filename) % 1000
            file_stack.push(file_id)
            print(f"    Opened {filename}")
        elif action == "get_recent":
            file_id = file_stack.pop()
            # Reverse lookup (simplified)
            filenames = ["document.txt", "presentation.ppt", "spreadsheet.xls"]
            filename = filenames[file_id % len(filenames)]
            print(f"    Most frequent recent: {filename}")
    
    # Application 2: Task priority system
    print(f"\n2. Task Priority System (Frequency-based):")
    task_stack = FreqStackOptimal()
    
    task_requests = [
        ("request", "backup_database"),
        ("request", "send_email"),
        ("request", "backup_database"),  # Higher priority due to frequency
        ("request", "generate_report"),
        ("request", "send_email"),
        ("request", "backup_database"),  # Highest priority
        ("execute", None),  # Should execute backup_database
        ("execute", None),  # Should execute send_email
        ("execute", None),  # Should execute backup_database again
    ]
    
    task_names = ["backup_database", "send_email", "generate_report"]
    
    print("  Task scheduling simulation:")
    for action, task in task_requests:
        if action == "request":
            task_id = hash(task) % 100
            task_stack.push(task_id)
            print(f"    Requested: {task}")
        elif action == "execute":
            task_id = task_stack.pop()
            task_name = task_names[task_id % len(task_names)]
            print(f"    Executing: {task_name} (highest frequency)")
    
    # Application 3: Cache with frequency-based eviction
    print(f"\n3. Frequency-based Cache System:")
    cache_stack = FreqStackOptimal()
    
    cache_accesses = [
        ("access", "user_profile_123"),
        ("access", "product_catalog"),
        ("access", "user_profile_123"),  # More frequent
        ("access", "shopping_cart"),
        ("access", "product_catalog"),
        ("access", "user_profile_123"),  # Most frequent
        ("evict", None),  # Evict most frequent (user_profile_123)
        ("evict", None),  # Evict next most frequent
    ]
    
    cache_items = ["user_profile_123", "product_catalog", "shopping_cart"]
    
    print("  Cache access pattern:")
    for action, item in cache_accesses:
        if action == "access":
            item_id = hash(item) % 100
            cache_stack.push(item_id)
            print(f"    Accessed: {item}")
        elif action == "evict":
            item_id = cache_stack.pop()
            item_name = cache_items[item_id % len(cache_items)]
            print(f"    Evicted: {item_name} (most frequently accessed)")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Optimal", "O(1)", "O(1)", "O(n)", "Frequency groups with stacks"),
        ("Heap", "O(log n)", "O(log n)", "O(n)", "Max heap operations"),
        ("List", "O(1)", "O(n)", "O(n)", "Linear search for pop"),
        ("Multiple Stacks", "O(1)", "O(1)", "O(n)", "Similar to optimal"),
    ]
    
    print(f"{'Approach':<20} | {'Push':<10} | {'Pop':<10} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, push_time, pop_time, space, notes in approaches:
        print(f"{approach:<20} | {push_time:<10} | {pop_time:<10} | {space:<8} | {notes}")
    
    print(f"\nOptimal approach provides O(1) for both operations")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    freq_stack = FreqStackOptimal()
    
    edge_cases = [
        ("Single element", lambda: (freq_stack.push(1), freq_stack.pop())[1], 1),
        ("Same element multiple times", lambda: (
            freq_stack.push(2), freq_stack.push(2), freq_stack.push(2),
            freq_stack.pop(), freq_stack.pop(), freq_stack.pop()
        )[5], 2),
        ("Alternating elements", lambda: (
            freq_stack.push(3), freq_stack.push(4), freq_stack.push(3),
            freq_stack.pop()  # Should return 3 (higher frequency)
        )[3], 3),
    ]
    
    for description, operation, expected in edge_cases:
        try:
            # Reset stack for each test
            freq_stack = FreqStackOptimal()
            result = operation()
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | Result: {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def benchmark_implementations():
    """Benchmark different implementations"""
    import time
    import random
    
    implementations = [
        ("Optimal", FreqStackOptimal),
        ("Multiple Stacks", FreqStackMultipleStacks),
    ]
    
    n_operations = 10000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"Operations: {n_operations} mixed push/pop operations")
    
    for impl_name, impl_class in implementations:
        try:
            freq_stack = impl_class()
            
            start_time = time.time()
            
            # Mixed operations with random values
            for i in range(n_operations):
                if random.random() < 0.7:  # 70% push operations
                    freq_stack.push(random.randint(1, 100))
                else:  # 30% pop operations
                    try:
                        freq_stack.pop()
                    except:
                        pass  # Handle empty stack
            
            end_time = time.time()
            
            print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:20} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_freq_stack_implementations()
    demonstrate_freq_stack_mechanism()
    visualize_frequency_groups()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_implementations()

"""
Maximum Frequency Stack demonstrates advanced system design with
frequency-based data structures, including multiple implementation
approaches for efficient frequency tracking and stack operations.
"""
