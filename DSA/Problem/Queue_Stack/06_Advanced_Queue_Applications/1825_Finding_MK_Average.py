"""
1825. Finding MK Average - Multiple Approaches
Difficulty: Hard

You are given two integers, m and k. You need to implement the MKAverage class:

- MKAverage(int m, int k) Initializes the object with an empty stream and the two integers m and k.
- void addElement(int num) Inserts a new element num into the stream.
- int calculateMKAverage() Calculates and returns the MK average, rounded down to the nearest integer.

The MK average is calculated as follows:
1. If the stream contains fewer than m elements, return -1.
2. Take the last m elements of the stream.
3. Remove the smallest k elements and the largest k elements.
4. Return the average of the rest of the elements, rounded down to the nearest integer.
"""

from typing import List
import heapq
from collections import deque
import bisect

class MKAverageHeaps:
    """
    Approach 1: Three Heaps Implementation (Optimal)
    
    Use three heaps to maintain smallest k, middle, and largest k elements.
    
    Time: O(log m) for add, O(1) for calculate, Space: O(m)
    """
    
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.stream = deque()
        
        # Three heaps to maintain order
        self.small = []  # Max heap (negated) for smallest k elements
        self.middle = []  # Min heap for middle elements
        self.large = []  # Min heap for largest k elements
        
        self.middle_sum = 0  # Sum of middle elements
    
    def addElement(self, num: int) -> None:
        """Add element to stream"""
        self.stream.append(num)
        
        if len(self.stream) > self.m:
            # Remove oldest element
            removed = self.stream.popleft()
            self._remove_element(removed)
        
        # Add new element
        self._add_element(num)
        self._rebalance()
    
    def calculateMKAverage(self) -> int:
        """Calculate MK average"""
        if len(self.stream) < self.m:
            return -1
        
        middle_count = self.m - 2 * self.k
        return self.middle_sum // middle_count
    
    def _add_element(self, num: int) -> None:
        """Add element to appropriate heap"""
        if len(self.small) < self.k:
            heapq.heappush(self.small, -num)
        elif len(self.large) < self.k:
            heapq.heappush(self.large, num)
        else:
            # Add to middle
            heapq.heappush(self.middle, num)
            self.middle_sum += num
    
    def _remove_element(self, num: int) -> None:
        """Remove element from appropriate heap"""
        if self.small and -num in [-x for x in self.small]:
            self.small.remove(-num)
            heapq.heapify(self.small)
        elif self.large and num in self.large:
            self.large.remove(num)
            heapq.heapify(self.large)
        elif self.middle and num in self.middle:
            self.middle.remove(num)
            heapq.heapify(self.middle)
            self.middle_sum -= num
    
    def _rebalance(self) -> None:
        """Rebalance the three heaps"""
        # Move elements between heaps to maintain size constraints
        while len(self.small) > self.k:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.middle, val)
            self.middle_sum += val
        
        while len(self.large) > self.k:
            val = heapq.heappop(self.large)
            heapq.heappush(self.middle, val)
            self.middle_sum += val
        
        # Ensure small heap has k smallest elements
        while len(self.small) < self.k and self.middle:
            if -self.small[0] if self.small else float('inf') > self.middle[0]:
                val = heapq.heappop(self.middle)
                heapq.heappush(self.small, -val)
                self.middle_sum -= val
        
        # Ensure large heap has k largest elements
        while len(self.large) < self.k and self.middle:
            if self.large[0] if self.large else float('-inf') < self.middle[-1]:
                # Find and remove largest from middle
                max_val = max(self.middle)
                self.middle.remove(max_val)
                heapq.heapify(self.middle)
                heapq.heappush(self.large, max_val)
                self.middle_sum -= max_val


class MKAverageSortedList:
    """
    Approach 2: Sorted List Implementation
    
    Use sorted list to maintain elements in order.
    
    Time: O(m) for add, O(1) for calculate, Space: O(m)
    """
    
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.stream = deque()
        self.sorted_elements = []
    
    def addElement(self, num: int) -> None:
        """Add element to stream"""
        self.stream.append(num)
        
        if len(self.stream) > self.m:
            # Remove oldest element
            removed = self.stream.popleft()
            self.sorted_elements.remove(removed)
        
        # Insert new element in sorted order
        bisect.insort(self.sorted_elements, num)
    
    def calculateMKAverage(self) -> int:
        """Calculate MK average"""
        if len(self.stream) < self.m:
            return -1
        
        # Take middle elements (excluding smallest k and largest k)
        middle_elements = self.sorted_elements[self.k:-self.k]
        return sum(middle_elements) // len(middle_elements)


class MKAverageSimple:
    """
    Approach 3: Simple Implementation
    
    Sort elements each time for calculation.
    
    Time: O(m log m) for calculate, O(1) for add, Space: O(m)
    """
    
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.stream = deque()
    
    def addElement(self, num: int) -> None:
        """Add element to stream"""
        self.stream.append(num)
        
        if len(self.stream) > self.m:
            self.stream.popleft()
    
    def calculateMKAverage(self) -> int:
        """Calculate MK average"""
        if len(self.stream) < self.m:
            return -1
        
        # Sort current elements
        sorted_elements = sorted(self.stream)
        
        # Take middle elements
        middle_elements = sorted_elements[self.k:-self.k]
        return sum(middle_elements) // len(middle_elements)


def test_mk_average_implementations():
    """Test MK average implementations"""
    
    implementations = [
        ("Three Heaps", MKAverageHeaps),
        ("Sorted List", MKAverageSortedList),
        ("Simple", MKAverageSimple),
    ]
    
    test_cases = [
        {
            "m": 3, "k": 1,
            "operations": ["addElement", "addElement", "calculateMKAverage", "addElement", "calculateMKAverage", "addElement", "calculateMKAverage"],
            "values": [3, 1, None, 10, None, 5, None],
            "expected": [None, None, -1, None, 4, None, 5],
            "description": "Example 1"
        },
        {
            "m": 5, "k": 2,
            "operations": ["addElement", "addElement", "addElement", "addElement", "addElement", "calculateMKAverage"],
            "values": [1, 2, 3, 4, 5, None],
            "expected": [None, None, None, None, None, 3],
            "description": "Full window"
        },
        {
            "m": 4, "k": 1,
            "operations": ["addElement", "addElement", "addElement", "calculateMKAverage"],
            "values": [1, 2, 3, None],
            "expected": [None, None, None, -1],
            "description": "Insufficient elements"
        },
    ]
    
    print("=== Testing MK Average Implementations ===")
    
    for impl_name, impl_class in implementations:
        print(f"\n--- {impl_name} Implementation ---")
        
        for test_case in test_cases:
            try:
                mk_avg = impl_class(test_case["m"], test_case["k"])
                results = []
                
                for i, op in enumerate(test_case["operations"]):
                    if op == "addElement":
                        mk_avg.addElement(test_case["values"][i])
                        results.append(None)
                    elif op == "calculateMKAverage":
                        result = mk_avg.calculateMKAverage()
                        results.append(result)
                
                expected = test_case["expected"]
                status = "✓" if results == expected else "✗"
                
                print(f"  {test_case['description']:20} | {status} | {results}")
                if results != expected:
                    print(f"    Expected: {expected}")
                
            except Exception as e:
                print(f"  {test_case['description']:20} | ERROR: {str(e)[:40]}")


def demonstrate_mk_average_calculation():
    """Demonstrate MK average calculation step by step"""
    print("\n=== MK Average Calculation Step-by-Step Demo ===")
    
    m, k = 5, 1
    mk_avg = MKAverageSimple(m, k)
    
    elements = [1, 3, 2, 5, 4, 6, 7]
    
    print(f"m = {m}, k = {k}")
    print("MK Average = average of middle elements after removing {k} smallest and {k} largest")
    
    for i, elem in enumerate(elements):
        mk_avg.addElement(elem)
        
        print(f"\nStep {i+1}: Added {elem}")
        print(f"  Stream: {list(mk_avg.stream)}")
        
        if len(mk_avg.stream) >= m:
            sorted_elements = sorted(mk_avg.stream)
            print(f"  Sorted: {sorted_elements}")
            
            smallest_k = sorted_elements[:k]
            largest_k = sorted_elements[-k:] if k > 0 else []
            middle = sorted_elements[k:-k] if k > 0 else sorted_elements
            
            print(f"  Remove smallest {k}: {smallest_k}")
            print(f"  Remove largest {k}: {largest_k}")
            print(f"  Middle elements: {middle}")
            
            mk_average = mk_avg.calculateMKAverage()
            print(f"  MK Average: {sum(middle)} / {len(middle)} = {mk_average}")
        else:
            print(f"  Not enough elements (need {m})")


def visualize_sliding_window():
    """Visualize sliding window behavior"""
    print("\n=== Sliding Window Visualization ===")
    
    m, k = 4, 1
    elements = [10, 20, 30, 40, 50, 60]
    
    print(f"Window size m = {m}, remove k = {k} from each end")
    print("Elements:", elements)
    
    window = deque()
    
    for i, elem in enumerate(elements):
        window.append(elem)
        
        if len(window) > m:
            removed = window.popleft()
            print(f"\nStep {i+1}: Added {elem}, removed {removed}")
        else:
            print(f"\nStep {i+1}: Added {elem}")
        
        print(f"  Window: {list(window)}")
        
        if len(window) == m:
            sorted_window = sorted(window)
            middle = sorted_window[k:-k] if k > 0 else sorted_window
            avg = sum(middle) // len(middle)
            
            print(f"  Sorted: {sorted_window}")
            print(f"  Middle: {middle}")
            print(f"  Average: {avg}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Stock price analysis
    print("1. Stock Price Trend Analysis:")
    
    # Remove outliers from recent stock prices
    m, k = 10, 2  # Last 10 prices, remove 2 highest and 2 lowest
    stock_analyzer = MKAverageSortedList(m, k)
    
    stock_prices = [100, 102, 98, 105, 95, 110, 88, 103, 99, 107, 101, 104]
    
    print(f"  Analyzing last {m} prices, removing {k} outliers from each end")
    
    for i, price in enumerate(stock_prices):
        stock_analyzer.addElement(price)
        
        if i >= m - 1:  # Have enough data
            trend_price = stock_analyzer.calculateMKAverage()
            print(f"    Day {i+1}: Price ${price}, Trend ${trend_price}")
    
    # Application 2: Performance metrics
    print(f"\n2. Server Response Time Analysis:")
    
    # Remove extreme response times
    m, k = 6, 1  # Last 6 responses, remove 1 fastest and 1 slowest
    perf_analyzer = MKAverageSortedList(m, k)
    
    response_times = [120, 85, 200, 95, 150, 300, 110, 180]  # milliseconds
    
    print(f"  Analyzing last {m} response times, removing {k} extreme from each end")
    
    for i, time_ms in enumerate(response_times):
        perf_analyzer.addElement(time_ms)
        
        if i >= m - 1:
            avg_time = perf_analyzer.calculateMKAverage()
            print(f"    Request {i+1}: {time_ms}ms, Stable avg: {avg_time}ms")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Three Heaps", "O(log m)", "O(1)", "O(m)", "Complex but optimal"),
        ("Sorted List", "O(m)", "O(1)", "O(m)", "Simple insertion sort"),
        ("Simple", "O(1)", "O(m log m)", "O(m)", "Sort on each calculation"),
    ]
    
    print(f"{'Approach':<15} | {'Add':<12} | {'Calculate':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, add_time, calc_time, space, notes in approaches:
        print(f"{approach:<15} | {add_time:<12} | {calc_time:<12} | {space:<8} | {notes}")
    
    print(f"\nChoice depends on add vs calculate frequency")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ((3, 1), [1, 2], -1, "Insufficient elements"),
        ((3, 1), [5, 5, 5], 5, "All same elements"),
        ((4, 2), [1, 2, 3, 4], 2, "k removes all but middle"),
        ((5, 0), [1, 2, 3, 4, 5], 3, "k = 0, no removal"),
        ((2, 0), [10, 20], 15, "Small window, no removal"),
    ]
    
    for (m, k), elements, expected, description in edge_cases:
        try:
            mk_avg = MKAverageSimple(m, k)
            
            for elem in elements:
                mk_avg.addElement(elem)
            
            result = mk_avg.calculateMKAverage()
            status = "✓" if result == expected else "✗"
            
            print(f"{description:30} | {status} | m={m}, k={k}, elements={elements} -> {result}")
            
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def benchmark_implementations():
    """Benchmark different implementations"""
    import time
    import random
    
    implementations = [
        ("Sorted List", MKAverageSortedList),
        ("Simple", MKAverageSimple),
    ]
    
    m, k = 100, 10
    n_operations = 1000
    
    print(f"\n=== Performance Benchmark ===")
    print(f"m = {m}, k = {k}, operations = {n_operations}")
    
    for impl_name, impl_class in implementations:
        try:
            mk_avg = impl_class(m, k)
            
            start_time = time.time()
            
            # Mixed add and calculate operations
            for i in range(n_operations):
                if i % 10 == 0 and i >= m:
                    mk_avg.calculateMKAverage()
                else:
                    mk_avg.addElement(random.randint(1, 1000))
            
            end_time = time.time()
            
            print(f"{impl_name:15} | Time: {end_time - start_time:.4f}s")
            
        except Exception as e:
            print(f"{impl_name:15} | ERROR: {str(e)[:30]}")


if __name__ == "__main__":
    test_mk_average_implementations()
    demonstrate_mk_average_calculation()
    visualize_sliding_window()
    demonstrate_real_world_applications()
    analyze_time_complexity()
    test_edge_cases()
    benchmark_implementations()

"""
Finding MK Average demonstrates advanced queue applications for
streaming data analysis with outlier removal, including multiple
approaches for sliding window statistics and robust averaging.
"""
