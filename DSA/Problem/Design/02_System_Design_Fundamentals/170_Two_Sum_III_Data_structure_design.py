"""
170. Two Sum III - Data structure design - Multiple Approaches
Difficulty: Easy

Design a data structure that accepts a stream of integers and checks if it has a pair of integers that sum to a particular value.

Implement the TwoSum class:
- TwoSum() Initializes the TwoSum object.
- void add(int number) Adds number to the data structure.
- boolean find(int value) Returns true if there exists any pair of numbers whose sum is equal to value.
"""

from typing import Dict, Set, List
from collections import defaultdict
import bisect

class TwoSumHashMap:
    """
    Approach 1: HashMap for Frequency Counting
    
    Store numbers with their frequencies and check complements.
    
    Time Complexity:
    - add: O(1)
    - find: O(n) where n is number of unique values
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.num_counts = defaultdict(int)
    
    def add(self, number: int) -> None:
        self.num_counts[number] += 1
    
    def find(self, value: int) -> bool:
        for num in self.num_counts:
            complement = value - num
            
            if complement == num:
                # Need at least 2 occurrences of the same number
                if self.num_counts[num] >= 2:
                    return True
            else:
                # Different numbers
                if complement in self.num_counts:
                    return True
        
        return False

class TwoSumHashSet:
    """
    Approach 2: HashSet with All Possible Sums
    
    Precompute all possible sums and store in set.
    
    Time Complexity:
    - add: O(n) where n is current number of elements
    - find: O(1)
    
    Space Complexity: O(n²)
    """
    
    def __init__(self):
        self.numbers = []
        self.sums = set()
    
    def add(self, number: int) -> None:
        # Generate all sums with existing numbers
        for existing_num in self.numbers:
            self.sums.add(number + existing_num)
        
        self.numbers.append(number)
    
    def find(self, value: int) -> bool:
        return value in self.sums

class TwoSumSortedList:
    """
    Approach 3: Sorted List with Two Pointers
    
    Maintain sorted list and use two pointers for find operation.
    
    Time Complexity:
    - add: O(n) for insertion in sorted order
    - find: O(n) for two pointers scan
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.numbers = []
    
    def add(self, number: int) -> None:
        # Insert in sorted order
        bisect.insort(self.numbers, number)
    
    def find(self, value: int) -> bool:
        if len(self.numbers) < 2:
            return False
        
        left, right = 0, len(self.numbers) - 1
        
        while left < right:
            current_sum = self.numbers[left] + self.numbers[right]
            
            if current_sum == value:
                return True
            elif current_sum < value:
                left += 1
            else:
                right -= 1
        
        return False

class TwoSumHybrid:
    """
    Approach 4: Hybrid Approach - Balance Add and Find
    
    Use list for storage and set for fast complement lookup during find.
    
    Time Complexity:
    - add: O(1)
    - find: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.numbers = []
        self.num_set = set()
    
    def add(self, number: int) -> None:
        self.numbers.append(number)
        self.num_set.add(number)
    
    def find(self, value: int) -> bool:
        seen = set()
        
        for num in self.numbers:
            complement = value - num
            
            if complement in seen:
                return True
            
            seen.add(num)
        
        return False

class TwoSumOptimized:
    """
    Approach 5: Optimized with Early Termination
    
    Enhanced version with optimizations and duplicate handling.
    
    Time Complexity:
    - add: O(1)
    - find: O(n)
    
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.num_counts = defaultdict(int)
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def add(self, number: int) -> None:
        self.num_counts[number] += 1
        self.min_val = min(self.min_val, number)
        self.max_val = max(self.max_val, number)
    
    def find(self, value: int) -> bool:
        # Early termination: if value is impossible
        if value < 2 * self.min_val or value > 2 * self.max_val:
            return False
        
        for num in self.num_counts:
            complement = value - num
            
            # Early termination: complement out of range
            if complement < self.min_val or complement > self.max_val:
                continue
            
            if complement == num:
                if self.num_counts[num] >= 2:
                    return True
            else:
                if complement in self.num_counts:
                    return True
        
        return False


def test_two_sum_basic():
    """Test basic TwoSum operations"""
    print("=== Testing Basic TwoSum Operations ===")
    
    implementations = [
        ("HashMap Frequency", TwoSumHashMap),
        ("HashSet All Sums", TwoSumHashSet),
        ("Sorted List", TwoSumSortedList),
        ("Hybrid Approach", TwoSumHybrid),
        ("Optimized", TwoSumOptimized)
    ]
    
    for name, TwoSumClass in implementations:
        print(f"\n{name}:")
        
        two_sum = TwoSumClass()
        
        # Test sequence: add some numbers, then test finds
        numbers = [1, 3, 5]
        for num in numbers:
            two_sum.add(num)
            print(f"  add({num})")
        
        # Test various find operations
        test_values = [4, 6, 7, 8, 2]
        for val in test_values:
            result = two_sum.find(val)
            print(f"  find({val}): {result}")

def test_two_sum_edge_cases():
    """Test TwoSum edge cases"""
    print("\n=== Testing TwoSum Edge Cases ===")
    
    two_sum = TwoSumHashMap()
    
    # Test with empty data structure
    print("Empty data structure:")
    print(f"  find(0): {two_sum.find(0)}")
    print(f"  find(5): {two_sum.find(5)}")
    
    # Test with single element
    print(f"\nSingle element:")
    two_sum.add(5)
    print(f"  add(5), find(10): {two_sum.find(10)}")
    print(f"  find(5): {two_sum.find(5)}")
    
    # Test with duplicates
    print(f"\nDuplicate elements:")
    two_sum.add(5)  # Now we have two 5s
    print(f"  add(5) again, find(10): {two_sum.find(10)}")  # 5 + 5 = 10
    
    # Test with negative numbers
    print(f"\nNegative numbers:")
    two_sum.add(-3)
    two_sum.add(8)
    print(f"  add(-3), add(8)")
    print(f"  find(5): {two_sum.find(5)}")   # 8 + (-3) = 5
    print(f"  find(2): {two_sum.find(2)}")   # 5 + (-3) = 2

def test_two_sum_duplicates():
    """Test handling of duplicate numbers"""
    print("\n=== Testing Duplicate Handling ===")
    
    two_sum = TwoSumOptimized()
    
    # Add duplicates and test various sums
    sequence = [1, 1, 2, 2, 3, 3]
    
    print("Adding sequence with duplicates:")
    for num in sequence:
        two_sum.add(num)
        print(f"  add({num})")
    
    # Test sums involving duplicates
    test_cases = [
        (2, "1+1"),
        (3, "1+2"), 
        (4, "2+2 or 1+3"),
        (5, "2+3"),
        (6, "3+3"),
        (7, "impossible")
    ]
    
    for target, explanation in test_cases:
        result = two_sum.find(target)
        print(f"  find({target}): {result} ({explanation})")

def test_two_sum_performance():
    """Test TwoSum performance"""
    print("\n=== Testing TwoSum Performance ===")
    
    import time
    
    implementations = [
        ("HashMap Frequency", TwoSumHashMap),
        ("Sorted List", TwoSumSortedList),
        ("Hybrid Approach", TwoSumHybrid),
        ("Optimized", TwoSumOptimized)
    ]
    
    num_elements = 1000
    num_queries = 100
    
    for name, TwoSumClass in implementations:
        two_sum = TwoSumClass()
        
        # Test add performance
        start_time = time.time()
        for i in range(num_elements):
            two_sum.add(i)
        add_time = (time.time() - start_time) * 1000
        
        # Test find performance
        start_time = time.time()
        for i in range(num_queries):
            target = i * 2  # Should find pairs like (0,0), (1,1), etc.
            two_sum.find(target)
        find_time = (time.time() - start_time) * 1000
        
        print(f"  {name}:")
        print(f"    Add {num_elements}: {add_time:.2f}ms")
        print(f"    Find {num_queries}: {find_time:.2f}ms")

def test_memory_efficiency():
    """Test memory efficiency of different approaches"""
    print("\n=== Testing Memory Efficiency ===")
    
    # Test with different approaches for memory usage
    test_size = 100
    
    implementations = [
        ("HashMap Frequency", TwoSumHashMap),
        ("HashSet All Sums", TwoSumHashSet),
        ("Hybrid Approach", TwoSumHybrid)
    ]
    
    for name, TwoSumClass in implementations:
        two_sum = TwoSumClass()
        
        # Add elements
        for i in range(test_size):
            two_sum.add(i)
        
        # Estimate memory usage (simplified)
        if hasattr(two_sum, 'num_counts'):
            memory_estimate = len(two_sum.num_counts)
        elif hasattr(two_sum, 'sums'):
            memory_estimate = len(two_sum.sums)
        elif hasattr(two_sum, 'numbers'):
            memory_estimate = len(two_sum.numbers)
        else:
            memory_estimate = test_size
        
        print(f"  {name}: ~{memory_estimate} units for {test_size} elements")

def test_streaming_scenario():
    """Test streaming data scenario"""
    print("\n=== Testing Streaming Scenario ===")
    
    two_sum = TwoSumOptimized()
    
    # Simulate streaming data with interleaved add/find operations
    print("Streaming data simulation:")
    
    stream_data = [
        ("add", 1), ("add", 2), ("find", 3),
        ("add", 4), ("find", 5), ("find", 6),
        ("add", 3), ("find", 7), ("add", 5),
        ("find", 8), ("find", 9), ("find", 10)
    ]
    
    for operation, value in stream_data:
        if operation == "add":
            two_sum.add(value)
            print(f"  Stream: add({value})")
        else:  # find
            result = two_sum.find(value)
            print(f"  Query: find({value}) = {result}")

def demonstrate_applications():
    """Demonstrate real-world applications"""
    print("\n=== Demonstrating Applications ===")
    
    # Application 1: Transaction monitoring
    print("Application 1: Transaction Fraud Detection")
    fraud_detector = TwoSumHashMap()
    
    # Add recent transaction amounts
    transactions = [100, 250, 75, 500, 25, 150]
    target_amount = 525  # Suspicious round number
    
    print(f"  Monitoring for suspicious pairs totaling ${target_amount}")
    
    for amount in transactions:
        fraud_detector.add(amount)
        
        # Check if this amount can pair with previous transactions
        if fraud_detector.find(target_amount):
            print(f"  ALERT: Transaction ${amount} can pair to reach ${target_amount}")
        else:
            print(f"  Transaction ${amount} added, no suspicious pairs yet")
    
    # Application 2: Complement pair finding in datasets
    print(f"\nApplication 2: Dataset Complement Analysis")
    data_analyzer = TwoSumSortedList()
    
    dataset = [10, 15, 3, 7, 8, 12, 20]
    target_sum = 22
    
    print(f"  Dataset: {dataset}")
    print(f"  Looking for pairs that sum to {target_sum}")
    
    for value in dataset:
        data_analyzer.add(value)
    
    if data_analyzer.find(target_sum):
        print(f"  ✓ Found pair(s) that sum to {target_sum}")
    else:
        print(f"  ✗ No pairs sum to {target_sum}")
    
    # Application 3: Chemistry compound analysis
    print(f"\nApplication 3: Chemical Compound Mass Analysis")
    mass_analyzer = TwoSumHybrid()
    
    atomic_masses = [1, 12, 14, 16, 32]  # H, C, N, O, S
    target_mass = 28  # Looking for combinations with mass 28
    
    print(f"  Atomic masses: {atomic_masses}")
    print(f"  Target compound mass: {target_mass}")
    
    for mass in atomic_masses:
        mass_analyzer.add(mass)
    
    result = mass_analyzer.find(target_mass)
    print(f"  Can form compound with mass {target_mass}: {result}")

def test_large_scale_operations():
    """Test large scale operations"""
    print("\n=== Testing Large Scale Operations ===")
    
    import random
    
    two_sum = TwoSumOptimized()
    
    # Add large number of elements
    large_size = 5000
    print(f"Adding {large_size} random elements...")
    
    for _ in range(large_size):
        two_sum.add(random.randint(1, 10000))
    
    # Test multiple find operations
    print("Testing multiple find operations...")
    found_count = 0
    total_tests = 100
    
    for _ in range(total_tests):
        target = random.randint(2, 20000)
        if two_sum.find(target):
            found_count += 1
    
    success_rate = (found_count / total_tests) * 100
    print(f"  Found pairs in {found_count}/{total_tests} queries ({success_rate:.1f}% success rate)")

def benchmark_add_vs_find_tradeoff():
    """Benchmark the add vs find performance tradeoff"""
    print("\n=== Benchmarking Add vs Find Tradeoff ===")
    
    import time
    
    # Compare approaches with different add/find ratios
    test_cases = [
        ("Frequent adds, few finds", 1000, 10),
        ("Balanced add/find", 500, 500),
        ("Few adds, frequent finds", 100, 1000)
    ]
    
    implementations = [
        ("HashMap (O(1) add, O(n) find)", TwoSumHashMap),
        ("HashSet (O(n) add, O(1) find)", TwoSumHashSet)
    ]
    
    for scenario_name, num_adds, num_finds in test_cases:
        print(f"\n{scenario_name}:")
        
        for impl_name, TwoSumClass in implementations:
            two_sum = TwoSumClass()
            
            # Time add operations
            start_time = time.time()
            for i in range(num_adds):
                two_sum.add(i)
            add_time = (time.time() - start_time) * 1000
            
            # Time find operations
            start_time = time.time()
            for i in range(num_finds):
                two_sum.find(i * 2)
            find_time = (time.time() - start_time) * 1000
            
            total_time = add_time + find_time
            print(f"  {impl_name}: {total_time:.2f}ms total (add: {add_time:.2f}ms, find: {find_time:.2f}ms)")

def test_boundary_conditions():
    """Test boundary conditions and edge cases"""
    print("\n=== Testing Boundary Conditions ===")
    
    two_sum = TwoSumOptimized()
    
    # Test with extreme values
    print("Testing extreme values:")
    extreme_values = [-1000000, 0, 1000000]
    
    for val in extreme_values:
        two_sum.add(val)
        print(f"  add({val})")
    
    # Test boundary sums
    boundary_tests = [
        (-2000000, "sum of two minimums"),
        (0, "negative + positive"),
        (2000000, "sum of two maximums"),
        (1, "impossible small sum"),
        (-999999, "near minimum range")
    ]
    
    for target, description in boundary_tests:
        result = two_sum.find(target)
        print(f"  find({target}): {result} ({description})")

if __name__ == "__main__":
    test_two_sum_basic()
    test_two_sum_edge_cases()
    test_two_sum_duplicates()
    test_two_sum_performance()
    test_memory_efficiency()
    test_streaming_scenario()
    demonstrate_applications()
    test_large_scale_operations()
    benchmark_add_vs_find_tradeoff()
    test_boundary_conditions()

"""
TwoSum Data Structure Design demonstrates key concepts:

Core Approaches:
1. HashMap Frequency - Store counts, O(1) add, O(n) find
2. HashSet All Sums - Precompute sums, O(n) add, O(1) find
3. Sorted List - Two pointers technique, O(n) add, O(n) find
4. Hybrid - List + Set combination for balanced performance
5. Optimized - Early termination and range checking

Key Design Trade-offs:
- Add performance vs Find performance
- Memory usage vs Query speed
- Duplicate handling strategies
- Range-based optimizations

Performance Characteristics:
- HashMap: Best for add-heavy workloads
- HashSet: Best for find-heavy workloads
- Sorted List: Balanced memory usage
- Optimized: Best overall with early termination

Real-world Applications:
- Fraud detection in financial transactions
- Complement analysis in scientific datasets
- Chemical compound mass calculations
- Pair finding in recommendation systems
- Streaming data analysis

The choice of approach depends on the expected
ratio of add to find operations in the use case.
"""
