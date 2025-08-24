"""
1352. Product of the Last K Numbers - Multiple Approaches
Difficulty: Medium

Design an algorithm that accepts a stream of integers and retrieves the product of the last k integers of the stream.

Implement the ProductOfNumbers class:
- ProductOfNumbers() Initializes the object with an empty stream.
- void add(int num) Adds the integer num to the end of the stream.
- int getProduct(int k) Returns the product of the last k numbers in the current list. You can assume that always the current list has at least k numbers.

The test cases are generated so that, at any time, the product of any contiguous sequence of numbers will fit into a single 32-bit integer without overflowing.
"""

from typing import List
from collections import deque

class ProductOfNumbersDeque:
    """
    Approach 1: Deque with Direct Calculation
    
    Use deque to store numbers and calculate product directly.
    
    Time: O(k) for getProduct, O(1) for add, Space: O(n)
    """
    
    def __init__(self):
        self.nums = deque()
    
    def add(self, num: int) -> None:
        """Add number to stream"""
        self.nums.append(num)
    
    def getProduct(self, k: int) -> int:
        """Get product of last k numbers"""
        product = 1
        
        # Calculate product of last k numbers
        for i in range(len(self.nums) - k, len(self.nums)):
            product *= self.nums[i]
        
        return product


class ProductOfNumbersPrefixProduct:
    """
    Approach 2: Prefix Product Array
    
    Use prefix products to calculate range products efficiently.
    
    Time: O(1) for getProduct, O(1) for add, Space: O(n)
    """
    
    def __init__(self):
        self.prefix_products = [1]  # prefix_products[i] = product of first i numbers
        self.nums = []
    
    def add(self, num: int) -> None:
        """Add number to stream"""
        self.nums.append(num)
        # Update prefix product
        self.prefix_products.append(self.prefix_products[-1] * num)
    
    def getProduct(self, k: int) -> int:
        """Get product of last k numbers using prefix products"""
        n = len(self.nums)
        
        # Product of last k numbers = prefix_products[n] / prefix_products[n-k]
        return self.prefix_products[n] // self.prefix_products[n - k]


class ProductOfNumbersZeroHandling:
    """
    Approach 3: Zero Handling with Prefix Products
    
    Handle zeros specially to avoid division by zero.
    
    Time: O(1) for getProduct (amortized), O(1) for add, Space: O(n)
    """
    
    def __init__(self):
        self.prefix_products = [1]
        self.zero_positions = []  # Track positions of zeros
        self.nums = []
    
    def add(self, num: int) -> None:
        """Add number to stream"""
        self.nums.append(num)
        
        if num == 0:
            self.zero_positions.append(len(self.nums) - 1)
            # Don't multiply by zero in prefix product
            self.prefix_products.append(self.prefix_products[-1])
        else:
            self.prefix_products.append(self.prefix_products[-1] * num)
    
    def getProduct(self, k: int) -> int:
        """Get product of last k numbers handling zeros"""
        n = len(self.nums)
        start_idx = n - k
        
        # Check if there are any zeros in the range [start_idx, n-1]
        zeros_in_range = sum(1 for pos in self.zero_positions if start_idx <= pos < n)
        
        if zeros_in_range > 0:
            return 0
        
        # No zeros in range, use prefix products
        return self.prefix_products[n] // self.prefix_products[start_idx]


class ProductOfNumbersSlidingWindow:
    """
    Approach 4: Sliding Window with Product Tracking
    
    Maintain sliding window of products.
    
    Time: O(k) for getProduct, O(1) for add, Space: O(n)
    """
    
    def __init__(self):
        self.nums = []
        self.window_products = {}  # Cache products for different window sizes
    
    def add(self, num: int) -> None:
        """Add number to stream"""
        self.nums.append(num)
        # Clear cache as new number affects all products
        self.window_products.clear()
    
    def getProduct(self, k: int) -> int:
        """Get product of last k numbers with caching"""
        if k in self.window_products:
            return self.window_products[k]
        
        product = 1
        n = len(self.nums)
        
        for i in range(n - k, n):
            product *= self.nums[i]
        
        self.window_products[k] = product
        return product


class ProductOfNumbersSegmentTree:
    """
    Approach 5: Segment Tree for Range Products
    
    Use segment tree for efficient range product queries.
    
    Time: O(log n) for getProduct, O(log n) for add, Space: O(n)
    """
    
    def __init__(self):
        self.nums = []
        self.tree = [1]  # Segment tree for products
        self.size = 1
    
    def add(self, num: int) -> None:
        """Add number to stream"""
        self.nums.append(num)
        
        # Expand tree if needed
        while self.size < len(self.nums):
            self.tree.extend([1] * self.size)
            self.size *= 2
        
        # Update tree
        self._update(0, 0, self.size - 1, len(self.nums) - 1, num)
    
    def _update(self, node: int, start: int, end: int, idx: int, val: int) -> None:
        """Update segment tree"""
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node + 1, start, mid, idx, val)
            else:
                self._update(2 * node + 2, mid + 1, end, idx, val)
            
            left_child = self.tree[2 * node + 1] if 2 * node + 1 < len(self.tree) else 1
            right_child = self.tree[2 * node + 2] if 2 * node + 2 < len(self.tree) else 1
            self.tree[node] = left_child * right_child
    
    def getProduct(self, k: int) -> int:
        """Get product of last k numbers"""
        n = len(self.nums)
        if k == 0:
            return 1
        
        return self._query(0, 0, self.size - 1, n - k, n - 1)
    
    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Query range product from segment tree"""
        if r < start or end < l:
            return 1
        
        if l <= start and end <= r:
            return self.tree[node]
        
        mid = (start + end) // 2
        left_product = self._query(2 * node + 1, start, mid, l, r)
        right_product = self._query(2 * node + 2, mid + 1, end, l, r)
        
        return left_product * right_product


class ProductOfNumbersOptimized:
    """
    Approach 6: Optimized with Zero Tracking
    
    Optimized approach handling zeros and using prefix products.
    
    Time: O(1) for both operations, Space: O(n)
    """
    
    def __init__(self):
        self.nums = []
        self.prefix_products = [1]
        self.zero_count = 0
        self.last_zero_index = -1
    
    def add(self, num: int) -> None:
        """Add number to stream"""
        self.nums.append(num)
        
        if num == 0:
            self.zero_count += 1
            self.last_zero_index = len(self.nums) - 1
            self.prefix_products.append(self.prefix_products[-1])
        else:
            self.prefix_products.append(self.prefix_products[-1] * num)
    
    def getProduct(self, k: int) -> int:
        """Get product of last k numbers"""
        n = len(self.nums)
        start_idx = n - k
        
        # Check if there's a zero in the range
        if self.last_zero_index >= start_idx:
            return 0
        
        # No zeros in range
        return self.prefix_products[n] // self.prefix_products[start_idx]


def test_product_of_numbers_implementations():
    """Test product of numbers implementations"""
    
    implementations = [
        ("Deque", ProductOfNumbersDeque),
        ("Prefix Product", ProductOfNumbersPrefixProduct),
        ("Zero Handling", ProductOfNumbersZeroHandling),
        ("Sliding Window", ProductOfNumbersSlidingWindow),
        ("Segment Tree", ProductOfNumbersSegmentTree),
        ("Optimized", ProductOfNumbersOptimized),
    ]
    
    test_cases = [
        {
            "operations": ["add", "add", "add", "getProduct", "add", "getProduct", "getProduct"],
            "values": [3, 0, 2, 2, 4, 2, 3],
            "expected": [None, None, None, 0, None, 8, 0],
            "description": "Example with zero"
        },
        {
            "operations": ["add", "add", "add", "getProduct", "add", "getProduct"],
            "values": [1, 2, 3, 3, 4, 2],
            "expected": [None, None, None, 6, None, 12],
            "description": "No zeros"
        },
        {
            "operations": ["add", "getProduct", "add", "getProduct"],
            "values": [5, 1, 3, 2],
            "expected": [None, 5, None, 15],
            "description": "Simple case"
        },
    ]
    
    print("=== Testing Product of Numbers Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                obj = impl_class()
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "add":
                        obj.add(test_case["values"][i])
                        results.append(None)
                    elif op == "getProduct":
                        result = obj.getProduct(test_case["values"][i])
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:20} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:20} | ERROR: {str(e)[:40]}")


def demonstrate_prefix_product_approach():
    """Demonstrate prefix product approach"""
    print("\n=== Prefix Product Approach Demonstration ===")
    
    obj = ProductOfNumbersPrefixProduct()
    
    numbers = [3, 1, 2, 4, 5]
    
    print("Adding numbers and showing prefix products:")
    print(f"Initial prefix_products: {obj.prefix_products}")
    
    for num in numbers:
        obj.add(num)
        print(f"Added {num}: nums = {obj.nums}, prefix_products = {obj.prefix_products}")
    
    print(f"\nTesting getProduct operations:")
    
    for k in [1, 2, 3, 4, 5]:
        product = obj.getProduct(k)
        
        # Manual verification
        manual_product = 1
        for i in range(len(obj.nums) - k, len(obj.nums)):
            manual_product *= obj.nums[i]
        
        print(f"getProduct({k}): {product} (manual: {manual_product})")
        print(f"  Formula: prefix[{len(obj.nums)}] / prefix[{len(obj.nums) - k}] = {obj.prefix_products[len(obj.nums)]} / {obj.prefix_products[len(obj.nums) - k]} = {product}")


def demonstrate_zero_handling():
    """Demonstrate zero handling"""
    print("\n=== Zero Handling Demonstration ===")
    
    obj = ProductOfNumbersZeroHandling()
    
    numbers = [2, 3, 0, 4, 5]
    
    print("Adding numbers with zero:")
    
    for num in numbers:
        obj.add(num)
        print(f"Added {num}: nums = {obj.nums}")
        print(f"  Zero positions: {obj.zero_positions}")
        print(f"  Prefix products: {obj.prefix_products}")
    
    print(f"\nTesting getProduct with zero in range:")
    
    test_k_values = [1, 2, 3, 4, 5]
    
    for k in test_k_values:
        product = obj.getProduct(k)
        n = len(obj.nums)
        start_idx = n - k
        
        zeros_in_range = [pos for pos in obj.zero_positions if start_idx <= pos < n]
        
        print(f"getProduct({k}): range [{start_idx}, {n-1}]")
        print(f"  Numbers in range: {obj.nums[start_idx:n]}")
        print(f"  Zeros in range: {zeros_in_range}")
        print(f"  Result: {product}")


def visualize_sliding_window():
    """Visualize sliding window approach"""
    print("\n=== Sliding Window Visualization ===")
    
    numbers = [2, 3, 4, 1, 5]
    
    print(f"Numbers: {numbers}")
    print("Sliding window products:")
    
    for k in range(1, len(numbers) + 1):
        window = numbers[-k:]
        product = 1
        for num in window:
            product *= num
        
        print(f"  k={k}: window = {window}, product = {product}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Financial metrics
    print("1. Financial Moving Product Metrics:")
    obj = ProductOfNumbersOptimized()
    
    # Simulate daily growth rates (as percentages converted to decimals)
    growth_rates = [102, 98, 105, 100, 103]  # 2%, -2%, 5%, 0%, 3%
    
    print("  Daily growth rates (as percentages * 100):")
    for i, rate in enumerate(growth_rates):
        obj.add(rate)
        print(f"    Day {i+1}: {rate/100:.0%}")
    
    print("  Moving product metrics:")
    for k in [3, 5]:
        if len(obj.nums) >= k:
            product = obj.getProduct(k)
            # Convert back to percentage
            actual_product = product / (100 ** k)
            print(f"    Last {k} days compound growth: {actual_product:.4f}")
    
    # Application 2: Quality control
    print(f"\n2. Manufacturing Quality Scores:")
    quality_obj = ProductOfNumbersOptimized()
    
    quality_scores = [95, 98, 100, 92, 97]  # Quality percentages
    
    for score in quality_scores:
        quality_obj.add(score)
    
    print(f"  Quality scores: {quality_scores}")
    
    for k in [2, 3]:
        if len(quality_obj.nums) >= k:
            product = quality_obj.getProduct(k)
            avg_quality = (product / (100 ** (k-1))) ** (1/k)
            print(f"    Last {k} batches geometric mean: {avg_quality:.2f}%")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Deque", "O(k)", "O(1)", "O(n)", "Direct calculation"),
        ("Prefix Product", "O(1)", "O(1)", "O(n)", "Optimal for no zeros"),
        ("Zero Handling", "O(1)", "O(1)", "O(n)", "Handles zeros efficiently"),
        ("Sliding Window", "O(k)", "O(1)", "O(n)", "With caching optimization"),
        ("Segment Tree", "O(log n)", "O(log n)", "O(n)", "General range queries"),
        ("Optimized", "O(1)", "O(1)", "O(n)", "Best overall approach"),
    ]
    
    print(f"{'Approach':<20} | {'getProduct':<12} | {'add':<8} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, get_time, add_time, space, notes in approaches:
        print(f"{approach:<20} | {get_time:<12} | {add_time:<8} | {space:<8} | {notes}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    obj = ProductOfNumbersOptimized()
    
    edge_cases = [
        ("Single element", [5], [(1, 5)]),
        ("All zeros", [0, 0, 0], [(1, 0), (2, 0), (3, 0)]),
        ("Zero at start", [0, 2, 3], [(1, 3), (2, 6), (3, 0)]),
        ("Zero at end", [2, 3, 0], [(1, 0), (2, 0), (3, 0)]),
        ("Large numbers", [100, 200, 300], [(1, 300), (2, 60000), (3, 6000000)]),
        ("Ones only", [1, 1, 1, 1], [(1, 1), (2, 1), (3, 1), (4, 1)]),
    ]
    
    for description, numbers, queries in edge_cases:
        obj = ProductOfNumbersOptimized()
        
        print(f"\n{description}:")
        print(f"  Numbers: {numbers}")
        
        for num in numbers:
            obj.add(num)
        
        for k, expected in queries:
            try:
                result = obj.getProduct(k)
                status = "✓" if result == expected else "✗"
                print(f"    getProduct({k}): {result} (expected: {expected}) {status}")
            except Exception as e:
                print(f"    getProduct({k}): ERROR - {str(e)[:30]}")


if __name__ == "__main__":
    test_product_of_numbers_implementations()
    demonstrate_prefix_product_approach()
    demonstrate_zero_handling()
    visualize_sliding_window()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()

"""
Product of the Last K Numbers demonstrates advanced queue applications
for streaming data analysis, including prefix products, zero handling,
and multiple optimization strategies for range product queries.
"""
