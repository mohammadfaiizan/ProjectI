"""
703. Kth Largest Element in a Stream - Multiple Approaches
Difficulty: Easy

Design a class to find the kth largest element in a stream. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Implement KthLargest class:
- KthLargest(int k, int[] nums) Initializes the object with the integer k and the stream of integers nums.
- int add(int val) Appends the integer val to the stream and returns the element representing the kth largest element in the stream.
"""

from typing import List
import heapq

class KthLargestMinHeap:
    """
    Approach 1: Min Heap (Optimal)
    
    Use min heap of size k to maintain k largest elements.
    
    Time: O(log k) for add, Space: O(k)
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = []
        
        # Add all initial numbers
        for num in nums:
            self.add(num)
    
    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        
        # Keep only k largest elements
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        
        return self.heap[0]


class KthLargestSortedList:
    """
    Approach 2: Sorted List
    
    Maintain sorted list and return kth largest.
    
    Time: O(n) for add, Space: O(n)
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = sorted(nums, reverse=True)
    
    def add(self, val: int) -> int:
        # Binary search insertion
        left, right = 0, len(self.nums)
        
        while left < right:
            mid = (left + right) // 2
            if self.nums[mid] > val:
                left = mid + 1
            else:
                right = mid
        
        self.nums.insert(left, val)
        
        # Return kth largest (1-indexed)
        return self.nums[self.k - 1] if len(self.nums) >= self.k else self.nums[-1]


class KthLargestBruteForce:
    """
    Approach 3: Brute Force
    
    Sort entire array each time.
    
    Time: O(n log n) for add, Space: O(n)
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums[:]
    
    def add(self, val: int) -> int:
        self.nums.append(val)
        self.nums.sort(reverse=True)
        return self.nums[self.k - 1]


class KthLargestMaxHeap:
    """
    Approach 4: Max Heap with Size Limit
    
    Use max heap and extract k elements each time.
    
    Time: O(k log n) for add, Space: O(n)
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums[:]
    
    def add(self, val: int) -> int:
        self.nums.append(val)
        
        # Create max heap (negate values)
        heap = [-num for num in self.nums]
        heapq.heapify(heap)
        
        # Extract k-1 elements
        for _ in range(self.k - 1):
            if heap:
                heapq.heappop(heap)
        
        return -heap[0] if heap else float('-inf')


class KthLargestQuickSelect:
    """
    Approach 5: Quick Select
    
    Use quick select to find kth largest.
    
    Time: O(n) average for add, Space: O(n)
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.nums = nums[:]
    
    def add(self, val: int) -> int:
        self.nums.append(val)
        return self.quickselect(self.nums[:], 0, len(self.nums) - 1, self.k - 1)
    
    def quickselect(self, nums: List[int], left: int, right: int, k: int) -> int:
        """Find kth largest element using quickselect"""
        if left == right:
            return nums[left]
        
        # Partition around pivot
        pivot_idx = self.partition(nums, left, right)
        
        if k == pivot_idx:
            return nums[k]
        elif k < pivot_idx:
            return self.quickselect(nums, left, pivot_idx - 1, k)
        else:
            return self.quickselect(nums, pivot_idx + 1, right, k)
    
    def partition(self, nums: List[int], left: int, right: int) -> int:
        """Partition array around pivot (for descending order)"""
        pivot = nums[right]
        i = left
        
        for j in range(left, right):
            if nums[j] >= pivot:  # Descending order
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        
        nums[i], nums[right] = nums[right], nums[i]
        return i


class KthLargestBST:
    """
    Approach 6: Binary Search Tree
    
    Use BST to maintain sorted order.
    
    Time: O(log n) average for add, Space: O(n)
    """
    
    class TreeNode:
        def __init__(self, val: int):
            self.val = val
            self.count = 1  # Count of this value
            self.left = None
            self.right = None
            self.size = 1  # Size of subtree
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.root = None
        
        for num in nums:
            self.root = self.insert(self.root, num)
    
    def add(self, val: int) -> int:
        self.root = self.insert(self.root, val)
        return self.find_kth_largest(self.root, self.k)
    
    def insert(self, node, val: int):
        """Insert value into BST"""
        if not node:
            return self.TreeNode(val)
        
        if val == node.val:
            node.count += 1
        elif val > node.val:
            node.left = self.insert(node.left, val)
        else:
            node.right = self.insert(node.right, val)
        
        # Update size
        node.size = node.count
        if node.left:
            node.size += node.left.size
        if node.right:
            node.size += node.right.size
        
        return node
    
    def find_kth_largest(self, node, k: int) -> int:
        """Find kth largest element in BST"""
        if not node:
            return float('-inf')
        
        left_size = node.left.size if node.left else 0
        
        if k <= left_size:
            return self.find_kth_largest(node.left, k)
        elif k <= left_size + node.count:
            return node.val
        else:
            return self.find_kth_largest(node.right, k - left_size - node.count)


def test_kth_largest_element_in_stream():
    """Test kth largest element in stream algorithms"""
    
    test_cases = [
        (3, [4, 5, 8, 2], [3, 5, 10, 9, 4], [4, 5, 5, 8, 8], "Example 1"),
        (1, [1, 2, 3], [4, 5], [4, 5], "k=1 case"),
        (2, [1], [2, 3, 4], [2, 2, 3], "Small initial array"),
        (4, [7, 10, 9, 3, 20, 15], [8, 5, 4, 1], [10, 10, 9, 9], "Large initial array"),
        (1, [], [1, 2, 3], [1, 2, 3], "Empty initial array"),
    ]
    
    implementations = [
        ("Min Heap", KthLargestMinHeap),
        ("Sorted List", KthLargestSortedList),
        ("Brute Force", KthLargestBruteForce),
        ("Max Heap", KthLargestMaxHeap),
        ("Quick Select", KthLargestQuickSelect),
        ("BST", KthLargestBST),
    ]
    
    print("=== Testing Kth Largest Element in Stream ===")
    
    for k, initial_nums, add_values, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"k: {k}")
        print(f"Initial: {initial_nums}")
        print(f"Add values: {add_values}")
        print(f"Expected: {expected}")
        
        for impl_name, impl_class in implementations:
            try:
                kth_largest = impl_class(k, initial_nums[:])
                results = []
                
                for val in add_values:
                    result = kth_largest.add(val)
                    results.append(result)
                
                status = "✓" if results == expected else "✗"
                print(f"{impl_name:15} | {status} | Results: {results}")
                
            except Exception as e:
                print(f"{impl_name:15} | ERROR: {str(e)[:40]}")


def demonstrate_min_heap_approach():
    """Demonstrate min heap approach step by step"""
    print("\n=== Min Heap Approach Step-by-Step Demo ===")
    
    k = 3
    initial_nums = [4, 5, 8, 2]
    add_values = [3, 5, 10, 9, 4]
    
    print(f"k = {k}")
    print(f"Initial numbers: {initial_nums}")
    
    kth_largest = KthLargestMinHeap(k, [])
    
    print(f"\nInitializing with {initial_nums}:")
    for num in initial_nums:
        result = kth_largest.add(num)
        print(f"  Add {num}: heap = {sorted(kth_largest.heap)}, kth largest = {result}")
    
    print(f"\nAdding stream values {add_values}:")
    for val in add_values:
        result = kth_largest.add(val)
        print(f"  Add {val}: heap = {sorted(kth_largest.heap)}, kth largest = {result}")


def visualize_heap_operations():
    """Visualize heap operations"""
    print("\n=== Heap Operations Visualization ===")
    
    k = 3
    print(f"Maintaining min heap of size {k} for {k}th largest element")
    print("Min heap property: smallest element at root")
    print("For kth largest: root of min heap is the kth largest element")
    
    kth_largest = KthLargestMinHeap(k, [])
    values = [4, 5, 8, 2, 3]
    
    for val in values:
        print(f"\nAdding {val}:")
        print(f"  Before: heap = {kth_largest.heap}")
        
        # Simulate the add operation
        heapq.heappush(kth_largest.heap, val)
        print(f"  After push: heap = {kth_largest.heap}")
        
        if len(kth_largest.heap) > k:
            removed = heapq.heappop(kth_largest.heap)
            print(f"  Removed {removed}: heap = {kth_largest.heap}")
        
        print(f"  {k}th largest = {kth_largest.heap[0] if kth_largest.heap else 'N/A'}")


def benchmark_kth_largest_implementations():
    """Benchmark different implementations"""
    import time
    import random
    
    implementations = [
        ("Min Heap", KthLargestMinHeap),
        ("Sorted List", KthLargestSortedList),
        ("Brute Force", KthLargestBruteForce),
        ("BST", KthLargestBST),
    ]
    
    # Test parameters
    k = 10
    initial_size = 100
    add_operations = 1000
    
    print(f"\n=== Kth Largest Performance Benchmark ===")
    print(f"k = {k}, initial size = {initial_size}, add operations = {add_operations}")
    
    # Generate test data
    initial_nums = [random.randint(1, 1000) for _ in range(initial_size)]
    add_values = [random.randint(1, 1000) for _ in range(add_operations)]
    
    for impl_name, impl_class in implementations:
        try:
            start_time = time.time()
            
            # Initialize
            kth_largest = impl_class(k, initial_nums[:])
            
            # Perform add operations
            for val in add_values:
                kth_largest.add(val)
            
            end_time = time.time()
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        (1, [], [1], [1], "k=1, empty initial"),
        (1, [1], [2], [2], "k=1, single initial"),
        (2, [1], [2, 3], [2, 2], "k=2, insufficient initial"),
        (3, [1, 2, 3], [0], [1], "Add smaller than all"),
        (3, [1, 2, 3], [4], [2], "Add larger than all"),
        (1, [5, 4, 3, 2, 1], [6], [6], "k=1, sorted desc"),
        (5, [1, 2, 3, 4, 5], [0], [1], "k=size, add minimum"),
    ]
    
    for k, initial, add_vals, expected, description in edge_cases:
        try:
            kth_largest = KthLargestMinHeap(k, initial)
            results = []
            
            for val in add_vals:
                results.append(kth_largest.add(val))
            
            status = "✓" if results == expected else "✗"
            print(f"{description:25} | {status} | k={k}, init={initial}, add={add_vals} -> {results}")
            
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_time_complexities():
    """Compare time complexities of different approaches"""
    print("\n=== Time Complexity Comparison ===")
    
    approaches = [
        ("Min Heap", "O(log k)", "O(k)", "Optimal for streaming"),
        ("Sorted List", "O(n)", "O(n)", "Binary search insertion"),
        ("Brute Force", "O(n log n)", "O(n)", "Sort entire array each time"),
        ("Max Heap", "O(k log n)", "O(n)", "Extract k elements each time"),
        ("Quick Select", "O(n)", "O(n)", "Average case, worst O(n²)"),
        ("BST", "O(log n)", "O(n)", "Average case, worst O(n)"),
    ]
    
    print(f"{'Approach':<15} | {'Add Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 65)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<15} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Live leaderboard
    print("1. Live Gaming Leaderboard - Track top K players:")
    k = 3
    leaderboard = KthLargestMinHeap(k, [])
    
    scores = [100, 150, 120, 180, 90, 200, 110]
    print(f"  Tracking top {k} scores")
    
    for score in scores:
        kth_score = leaderboard.add(score)
        print(f"    New score: {score} -> {k}th highest: {kth_score}")
    
    # Application 2: Stock price monitoring
    print(f"\n2. Stock Price Monitoring - Track {k}rd highest price:")
    stock_monitor = KthLargestMinHeap(k, [])
    
    prices = [50.5, 52.0, 48.5, 55.0, 51.5, 60.0, 49.0]
    for price in prices:
        kth_price = stock_monitor.add(int(price * 100))  # Convert to cents
        print(f"    Price: ${price:.2f} -> {k}rd highest: ${kth_price/100:.2f}")
    
    # Application 3: Server response time monitoring
    print(f"\n3. Server Response Time - Monitor {k}rd slowest response:")
    response_monitor = KthLargestMinHeap(k, [])
    
    response_times = [120, 85, 200, 95, 150, 300, 110]  # milliseconds
    for time_ms in response_times:
        kth_time = response_monitor.add(time_ms)
        print(f"    Response: {time_ms}ms -> {k}rd slowest: {kth_time}ms")


def demonstrate_heap_properties():
    """Demonstrate heap properties"""
    print("\n=== Heap Properties Demonstration ===")
    
    print("Min Heap Properties:")
    print("1. Parent ≤ Children (min heap property)")
    print("2. Complete binary tree (filled level by level)")
    print("3. Root contains minimum element")
    print("4. For kth largest: maintain heap of size k, root is kth largest")
    
    k = 4
    heap = []
    values = [10, 5, 15, 3, 8, 12, 20, 1]
    
    print(f"\nBuilding min heap of size {k}:")
    
    for val in values:
        print(f"\nAdding {val}:")
        heapq.heappush(heap, val)
        print(f"  Heap after push: {heap}")
        
        if len(heap) > k:
            removed = heapq.heappop(heap)
            print(f"  Removed {removed}: {heap}")
        
        print(f"  Current {k}th largest: {heap[0] if heap else 'N/A'}")
        
        # Show heap structure
        if heap:
            print(f"  Heap structure: root={heap[0]}")
            if len(heap) > 1:
                children = heap[1:3] if len(heap) > 2 else heap[1:2]
                print(f"    Children of root: {children}")


if __name__ == "__main__":
    test_kth_largest_element_in_stream()
    demonstrate_min_heap_approach()
    visualize_heap_operations()
    demonstrate_heap_properties()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_time_complexities()
    benchmark_kth_largest_implementations()

"""
Kth Largest Element in a Stream demonstrates priority queue applications
for streaming data problems, including multiple heap-based approaches
and real-world applications in leaderboards and monitoring systems.
"""
