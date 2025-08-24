"""
295. Find Median from Data Stream - Multiple Approaches
Difficulty: Hard

The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

Implement the MedianFinder class:
- MedianFinder() initializes the MedianFinder object.
- void addNum(int num) adds the integer num from the data stream to the data structure.
- double findMedian() returns the median of all elements so far.
"""

import heapq
from typing import List

class MedianFinderTwoHeaps:
    """
    Approach 1: Two Heaps (Optimal)
    
    Use max heap for smaller half and min heap for larger half.
    
    Time: O(log n) for add, O(1) for find, Space: O(n)
    """
    
    def __init__(self):
        self.small = []  # Max heap (negate values)
        self.large = []  # Min heap
    
    def addNum(self, num: int) -> None:
        """Add number to data structure"""
        # Add to appropriate heap
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)
        
        # Balance heaps
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        elif len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)
    
    def findMedian(self) -> float:
        """Find median of all elements"""
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        elif len(self.large) > len(self.small):
            return float(self.large[0])
        else:
            return (-self.small[0] + self.large[0]) / 2.0


class MedianFinderSortedList:
    """
    Approach 2: Sorted List
    
    Maintain sorted list and find median by indexing.
    
    Time: O(n) for add, O(1) for find, Space: O(n)
    """
    
    def __init__(self):
        self.nums = []
    
    def addNum(self, num: int) -> None:
        """Add number using binary search insertion"""
        left, right = 0, len(self.nums)
        
        while left < right:
            mid = (left + right) // 2
            if self.nums[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        self.nums.insert(left, num)
    
    def findMedian(self) -> float:
        """Find median from sorted list"""
        n = len(self.nums)
        if n % 2 == 1:
            return float(self.nums[n // 2])
        else:
            return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2.0


class MedianFinderBruteForce:
    """
    Approach 3: Brute Force
    
    Store all numbers and sort each time to find median.
    
    Time: O(n log n) for add, O(1) for find, Space: O(n)
    """
    
    def __init__(self):
        self.nums = []
    
    def addNum(self, num: int) -> None:
        """Add number to list"""
        self.nums.append(num)
    
    def findMedian(self) -> float:
        """Sort and find median"""
        sorted_nums = sorted(self.nums)
        n = len(sorted_nums)
        
        if n % 2 == 1:
            return float(sorted_nums[n // 2])
        else:
            return (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2.0


class MedianFinderBST:
    """
    Approach 4: Binary Search Tree
    
    Use BST to maintain sorted order.
    
    Time: O(log n) average for add/find, Space: O(n)
    """
    
    class TreeNode:
        def __init__(self, val: int):
            self.val = val
            self.count = 1
            self.left = None
            self.right = None
            self.size = 1  # Size of subtree
    
    def __init__(self):
        self.root = None
        self.total_count = 0
    
    def addNum(self, num: int) -> None:
        """Add number to BST"""
        self.root = self._insert(self.root, num)
        self.total_count += 1
    
    def _insert(self, node, val: int):
        """Insert value into BST"""
        if not node:
            return self.TreeNode(val)
        
        if val == node.val:
            node.count += 1
        elif val < node.val:
            node.left = self._insert(node.left, val)
        else:
            node.right = self._insert(node.right, val)
        
        # Update size
        node.size = node.count
        if node.left:
            node.size += node.left.size
        if node.right:
            node.size += node.right.size
        
        return node
    
    def findMedian(self) -> float:
        """Find median using BST"""
        if self.total_count % 2 == 1:
            # Odd count: find middle element
            return float(self._find_kth(self.root, self.total_count // 2 + 1))
        else:
            # Even count: average of two middle elements
            left_mid = self._find_kth(self.root, self.total_count // 2)
            right_mid = self._find_kth(self.root, self.total_count // 2 + 1)
            return (left_mid + right_mid) / 2.0
    
    def _find_kth(self, node, k: int) -> int:
        """Find kth smallest element in BST"""
        if not node:
            return 0
        
        left_size = node.left.size if node.left else 0
        
        if k <= left_size:
            return self._find_kth(node.left, k)
        elif k <= left_size + node.count:
            return node.val
        else:
            return self._find_kth(node.right, k - left_size - node.count)


class MedianFinderBucketSort:
    """
    Approach 5: Bucket Sort (for bounded input)
    
    Use bucket sort when input range is known and bounded.
    
    Time: O(1) for add, O(range) for find, Space: O(range)
    """
    
    def __init__(self, min_val: int = -100000, max_val: int = 100000):
        self.min_val = min_val
        self.max_val = max_val
        self.buckets = [0] * (max_val - min_val + 1)
        self.count = 0
    
    def addNum(self, num: int) -> None:
        """Add number to bucket"""
        if self.min_val <= num <= self.max_val:
            self.buckets[num - self.min_val] += 1
            self.count += 1
    
    def findMedian(self) -> float:
        """Find median using bucket counts"""
        if self.count == 0:
            return 0.0
        
        if self.count % 2 == 1:
            # Find middle element
            target = self.count // 2 + 1
            current_count = 0
            
            for i, bucket_count in enumerate(self.buckets):
                current_count += bucket_count
                if current_count >= target:
                    return float(i + self.min_val)
        else:
            # Find two middle elements
            target1 = self.count // 2
            target2 = self.count // 2 + 1
            current_count = 0
            median_vals = []
            
            for i, bucket_count in enumerate(self.buckets):
                current_count += bucket_count
                
                if current_count >= target1 and len(median_vals) == 0:
                    median_vals.append(i + self.min_val)
                
                if current_count >= target2:
                    median_vals.append(i + self.min_val)
                    break
            
            return sum(median_vals) / 2.0
        
        return 0.0


def test_median_finder_implementations():
    """Test median finder implementations"""
    
    implementations = [
        ("Two Heaps", MedianFinderTwoHeaps),
        ("Sorted List", MedianFinderSortedList),
        ("Brute Force", MedianFinderBruteForce),
        ("BST", MedianFinderBST),
    ]
    
    test_operations = [
        ("addNum", 1),
        ("addNum", 2),
        ("findMedian", None, 1.5),
        ("addNum", 3),
        ("findMedian", None, 2.0),
        ("addNum", 4),
        ("addNum", 5),
        ("findMedian", None, 3.0),
    ]
    
    print("=== Testing Median Finder Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} ---")
        
        try:
            finder = impl_class()
            
            for operation in test_operations:
                if operation[0] == "addNum":
                    finder.addNum(operation[1])
                    print(f"  addNum({operation[1]})")
                
                elif operation[0] == "findMedian":
                    result = finder.findMedian()
                    expected = operation[2]
                    status = "✓" if abs(result - expected) < 1e-9 else "✗"
                    print(f"  findMedian() -> {result} (expected: {expected}) {status}")
            
            print("  ✓ All operations completed")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")


def demonstrate_two_heaps_approach():
    """Demonstrate two heaps approach step by step"""
    print("\n=== Two Heaps Approach Step-by-Step Demo ===")
    
    finder = MedianFinderTwoHeaps()
    numbers = [1, 5, 2, 3, 4]
    
    print("Adding numbers and finding median:")
    print("Strategy: Keep smaller half in max heap, larger half in min heap")
    
    for num in numbers:
        print(f"\nAdding {num}:")
        finder.addNum(num)
        
        # Show heap states
        small_vals = [-x for x in finder.small]
        large_vals = list(finder.large)
        
        print(f"  Small heap (max): {sorted(small_vals, reverse=True)}")
        print(f"  Large heap (min): {sorted(large_vals)}")
        print(f"  Sizes: small={len(finder.small)}, large={len(finder.large)}")
        
        median = finder.findMedian()
        print(f"  Median: {median}")


def visualize_heap_balancing():
    """Visualize heap balancing process"""
    print("\n=== Heap Balancing Visualization ===")
    
    finder = MedianFinderTwoHeaps()
    
    print("Heap balancing rules:")
    print("1. Size difference between heaps should be at most 1")
    print("2. All elements in small heap ≤ all elements in large heap")
    print("3. If odd total count: one heap has one more element")
    print("4. If even total count: both heaps have equal size")
    
    numbers = [6, 10, 2, 6, 5, 0, 6, 3, 1, 0, 0]
    
    for i, num in enumerate(numbers):
        print(f"\nStep {i+1}: Adding {num}")
        
        # Show before state
        small_before = [-x for x in finder.small]
        large_before = list(finder.large)
        
        finder.addNum(num)
        
        # Show after state
        small_after = [-x for x in finder.small]
        large_after = list(finder.large)
        
        print(f"  Before: small={small_before}, large={large_before}")
        print(f"  After:  small={small_after}, large={large_after}")
        
        # Check balance
        size_diff = abs(len(finder.small) - len(finder.large))
        balanced = size_diff <= 1
        
        print(f"  Balanced: {'✓' if balanced else '✗'} (size diff: {size_diff})")
        print(f"  Median: {finder.findMedian()}")


def benchmark_median_finder_implementations():
    """Benchmark different implementations"""
    import time
    import random
    
    implementations = [
        ("Two Heaps", MedianFinderTwoHeaps),
        ("Sorted List", MedianFinderSortedList),
        ("Brute Force", MedianFinderBruteForce),
        ("BST", MedianFinderBST),
    ]
    
    n_operations = 1000
    
    print(f"\n=== Median Finder Performance Benchmark ===")
    print(f"Operations: {n_operations} addNum + {n_operations//10} findMedian")
    
    for impl_name, impl_class in implementations:
        try:
            finder = impl_class()
            
            start_time = time.time()
            
            # Add operations
            for i in range(n_operations):
                num = random.randint(1, 1000)
                finder.addNum(num)
                
                # Find median occasionally
                if i % 10 == 0:
                    finder.findMedian()
            
            end_time = time.time()
            
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    finder = MedianFinderTwoHeaps()
    
    edge_cases = [
        ("Single element", [5], 5.0),
        ("Two elements", [1, 2], 1.5),
        ("All same", [3, 3, 3], 3.0),
        ("Negative numbers", [-1, -2, -3], -2.0),
        ("Mixed signs", [-1, 0, 1], 0.0),
        ("Large numbers", [1000000, 999999], 999999.5),
    ]
    
    for description, numbers, expected in edge_cases:
        finder = MedianFinderTwoHeaps()
        
        for num in numbers:
            finder.addNum(num)
        
        result = finder.findMedian()
        status = "✓" if abs(result - expected) < 1e-9 else "✗"
        
        print(f"{description:20} | {status} | numbers: {numbers} -> {result}")


def compare_time_complexities():
    """Compare time complexities of different approaches"""
    print("\n=== Time Complexity Comparison ===")
    
    approaches = [
        ("Two Heaps", "O(log n)", "O(1)", "O(n)", "Optimal for streaming"),
        ("Sorted List", "O(n)", "O(1)", "O(n)", "Binary search insertion"),
        ("Brute Force", "O(n log n)", "O(1)", "O(n)", "Sort every time"),
        ("BST", "O(log n)", "O(log n)", "O(n)", "Average case, worst O(n)"),
        ("Bucket Sort", "O(1)", "O(range)", "O(range)", "For bounded input"),
    ]
    
    print(f"{'Approach':<15} | {'Add':<12} | {'Find':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, add_time, find_time, space, notes in approaches:
        print(f"{approach:<15} | {add_time:<12} | {find_time:<12} | {space:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Real-time analytics
    print("1. Real-time Website Response Time Monitoring:")
    finder = MedianFinderTwoHeaps()
    
    response_times = [120, 85, 200, 95, 150, 300, 110, 180, 90, 250]  # milliseconds
    
    print("  Response times (ms):")
    for i, time_ms in enumerate(response_times):
        finder.addNum(time_ms)
        median = finder.findMedian()
        print(f"    Request {i+1}: {time_ms}ms -> Median: {median}ms")
    
    # Application 2: Stock price analysis
    print(f"\n2. Stock Price Median Tracking:")
    price_finder = MedianFinderTwoHeaps()
    
    stock_prices = [100.5, 102.3, 98.7, 105.1, 99.8, 103.2, 101.0]  # Convert to cents
    
    for i, price in enumerate(stock_prices):
        price_cents = int(price * 100)
        price_finder.addNum(price_cents)
        median_cents = price_finder.findMedian()
        median_price = median_cents / 100
        
        print(f"    Day {i+1}: ${price:.2f} -> Median: ${median_price:.2f}")


if __name__ == "__main__":
    test_median_finder_implementations()
    demonstrate_two_heaps_approach()
    visualize_heap_balancing()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_time_complexities()
    benchmark_median_finder_implementations()

"""
Find Median from Data Stream demonstrates advanced heap applications
for streaming data analysis, including two-heap technique and multiple
approaches for dynamic median calculation with real-time constraints.
"""
