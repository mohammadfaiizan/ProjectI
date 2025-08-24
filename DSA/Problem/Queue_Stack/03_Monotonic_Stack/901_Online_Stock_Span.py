"""
901. Online Stock Span - Multiple Approaches
Difficulty: Medium

Design an algorithm that collects daily price quotes for some stock and returns the span of that stock's price for the current day.

The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backward) for which the price of the stock was less than or equal to today's price.

Implement the StockSpanner class:
- StockSpanner() Initializes the object of the class.
- int next(int price) Returns the span of the stock's price given that today's price is price.
"""

from typing import List

class StockSpanner1:
    """
    Approach 1: Monotonic Stack (Optimal)
    
    Use stack to track previous prices and their spans.
    
    Time: O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self.stack = []  # Store (price, span) pairs
    
    def next(self, price: int) -> int:
        span = 1
        
        # Pop elements with price <= current price
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        
        self.stack.append((price, span))
        return span


class StockSpanner2:
    """
    Approach 2: Brute Force with History
    
    Store all prices and calculate span by looking back.
    
    Time: O(n) per call, Space: O(n)
    """
    
    def __init__(self):
        self.prices = []
    
    def next(self, price: int) -> int:
        self.prices.append(price)
        span = 1
        
        # Look backward to count consecutive days
        for i in range(len(self.prices) - 2, -1, -1):
            if self.prices[i] <= price:
                span += 1
            else:
                break
        
        return span


class StockSpanner3:
    """
    Approach 3: Stack with Indices
    
    Store indices in stack instead of (price, span) pairs.
    
    Time: O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self.prices = []
        self.stack = []  # Store indices
    
    def next(self, price: int) -> int:
        self.prices.append(price)
        current_index = len(self.prices) - 1
        
        # Pop indices with smaller or equal prices
        while self.stack and self.prices[self.stack[-1]] <= price:
            self.stack.pop()
        
        # Calculate span
        span = current_index - (self.stack[-1] if self.stack else -1)
        
        self.stack.append(current_index)
        return span


class StockSpanner4:
    """
    Approach 4: Optimized with Previous Greater Element
    
    Track previous greater element for each price.
    
    Time: O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        self.prices = []
        self.spans = []
        self.stack = []  # Store indices
    
    def next(self, price: int) -> int:
        self.prices.append(price)
        current_index = len(self.prices) - 1
        
        # Find previous greater element
        while self.stack and self.prices[self.stack[-1]] <= price:
            self.stack.pop()
        
        # Calculate span based on previous greater element
        prev_greater = self.stack[-1] if self.stack else -1
        span = current_index - prev_greater
        
        self.spans.append(span)
        self.stack.append(current_index)
        
        return span


class StockSpanner5:
    """
    Approach 5: Segment Tree Approach
    
    Use segment tree for range maximum queries.
    
    Time: O(log n) per call, Space: O(n)
    """
    
    def __init__(self):
        self.prices = []
        self.tree = {}
    
    def _update(self, node: int, start: int, end: int, idx: int, val: int) -> None:
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self._update(2 * node, start, mid, idx, val)
            else:
                self._update(2 * node + 1, mid + 1, end, idx, val)
            
            left_max = self.tree.get(2 * node, 0)
            right_max = self.tree.get(2 * node + 1, 0)
            self.tree[node] = max(left_max, right_max)
    
    def _query(self, node: int, start: int, end: int, l: int, r: int) -> int:
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return self.tree.get(node, 0)
        
        mid = (start + end) // 2
        left_max = self._query(2 * node, start, mid, l, r)
        right_max = self._query(2 * node + 1, mid + 1, end, l, r)
        return max(left_max, right_max)
    
    def next(self, price: int) -> int:
        self.prices.append(price)
        current_index = len(self.prices) - 1
        
        # Update segment tree
        self._update(1, 0, 10000, current_index, price)
        
        # Binary search for span
        left, right = 0, current_index
        span = 1
        
        while left <= right:
            mid = (left + right) // 2
            max_in_range = self._query(1, 0, 10000, mid, current_index - 1)
            
            if max_in_range <= price:
                span = current_index - mid + 1
                right = mid - 1
            else:
                left = mid + 1
        
        return span


class StockSpanner6:
    """
    Approach 6: Deque-based Approach
    
    Use deque for efficient front/back operations.
    
    Time: O(1) amortized, Space: O(n)
    """
    
    def __init__(self):
        from collections import deque
        self.stack = deque()  # Store (price, span) pairs
    
    def next(self, price: int) -> int:
        span = 1
        
        # Pop elements with price <= current price
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        
        self.stack.append((price, span))
        return span


def test_stock_spanner_implementations():
    """Test all stock spanner implementations"""
    
    implementations = [
        ("Monotonic Stack", StockSpanner1),
        ("Brute Force", StockSpanner2),
        ("Stack with Indices", StockSpanner3),
        ("Previous Greater", StockSpanner4),
        ("Deque Approach", StockSpanner6),
    ]
    
    test_cases = [
        ([100, 80, 60, 70, 60, 75, 85], [1, 1, 1, 2, 1, 4, 6], "Example 1"),
        ([31, 41, 48, 59, 79], [1, 2, 3, 4, 5], "Increasing prices"),
        ([100, 90, 80, 70, 60], [1, 1, 1, 1, 1], "Decreasing prices"),
        ([50, 50, 50], [1, 2, 3], "Same prices"),
        ([10], [1], "Single price"),
        ([1, 2, 1, 3], [1, 2, 1, 4], "Mixed pattern"),
    ]
    
    print("=== Testing Stock Spanner Implementations ===")
    
    for prices, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Prices: {prices}")
        print(f"Expected spans: {expected}")
        
        for impl_name, impl_class in implementations:
            try:
                spanner = impl_class()
                results = []
                
                for price in prices:
                    span = spanner.next(price)
                    results.append(span)
                
                status = "✓" if results == expected else "✗"
                print(f"{impl_name:20} | {status} | Results: {results}")
            
            except Exception as e:
                print(f"{impl_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_stock_span_concept():
    """Demonstrate stock span concept"""
    print("\n=== Stock Span Concept Demonstration ===")
    
    prices = [100, 80, 60, 70, 60, 75, 85]
    print(f"Daily prices: {prices}")
    print("\nStock span calculation:")
    
    for i, price in enumerate(prices):
        print(f"\nDay {i+1}: Price = {price}")
        
        # Calculate span manually
        span = 1
        for j in range(i-1, -1, -1):
            if prices[j] <= price:
                span += 1
                print(f"  Day {j+1}: Price {prices[j]} <= {price} ✓")
            else:
                print(f"  Day {j+1}: Price {prices[j]} > {price} ✗ (stop)")
                break
        
        print(f"  Span: {span} days")


def demonstrate_stack_approach():
    """Demonstrate stack approach step by step"""
    print("\n=== Stack Approach Step-by-Step Demo ===")
    
    prices = [100, 80, 60, 70, 60, 75, 85]
    print(f"Processing prices: {prices}")
    
    stack = []  # (price, span) pairs
    
    for i, price in enumerate(prices):
        print(f"\nDay {i+1}: Processing price {price}")
        print(f"  Stack before: {stack}")
        
        span = 1
        popped = []
        
        # Pop elements with price <= current price
        while stack and stack[-1][0] <= price:
            popped_item = stack.pop()
            span += popped_item[1]
            popped.append(popped_item)
        
        if popped:
            print(f"  Popped: {popped}")
            print(f"  Added their spans to current span: {span}")
        
        stack.append((price, span))
        print(f"  Stack after: {stack}")
        print(f"  Span for day {i+1}: {span}")


def visualize_stock_spans():
    """Visualize stock spans"""
    print("\n=== Stock Spans Visualization ===")
    
    prices = [100, 80, 60, 70, 60, 75, 85]
    print(f"Prices: {prices}")
    
    # Calculate spans
    spanner = StockSpanner1()
    spans = []
    for price in prices:
        spans.append(spanner.next(price))
    
    print(f"Spans:  {spans}")
    print()
    
    # Create visualization
    max_price = max(prices)
    
    print("Price chart with spans:")
    for level in range(max_price, 0, -10):
        line = f"{level:3} |"
        for i, price in enumerate(prices):
            if price >= level:
                line += "██"
            else:
                line += "  "
        print(line)
    
    # Show spans
    print("    +" + "--" * len(prices))
    print("     " + "".join(f"{i+1:2}" for i in range(len(prices))))
    print("Span:" + "".join(f"{s:2}" for s in spans))
    
    # Show span ranges
    print("\nSpan ranges:")
    for i, span in enumerate(spans):
        start_day = i - span + 2
        end_day = i + 1
        price_range = prices[i-span+1:i+1]
        print(f"Day {i+1}: Span {span} covers days {start_day}-{end_day} with prices {price_range}")


def benchmark_stock_spanner():
    """Benchmark different approaches"""
    import time
    import random
    
    implementations = [
        ("Monotonic Stack", StockSpanner1),
        ("Brute Force", StockSpanner2),
        ("Stack with Indices", StockSpanner3),
        ("Previous Greater", StockSpanner4),
    ]
    
    # Test with different numbers of operations
    operation_counts = [100, 1000, 5000]
    
    print("\n=== Stock Spanner Performance Benchmark ===")
    
    for n_ops in operation_counts:
        print(f"\n--- {n_ops} Operations ---")
        
        # Generate random prices
        prices = [random.randint(1, 1000) for _ in range(n_ops)]
        
        for impl_name, impl_class in implementations:
            spanner = impl_class()
            
            start_time = time.time()
            
            try:
                for price in prices:
                    spanner.next(price)
                
                end_time = time.time()
                print(f"{impl_name:20} | Time: {end_time - start_time:.4f}s")
            except Exception as e:
                print(f"{impl_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    edge_cases = [
        ([1], [1], "Single price"),
        ([1, 1], [1, 2], "Two same prices"),
        ([1, 2, 3], [1, 2, 3], "Strictly increasing"),
        ([3, 2, 1], [1, 1, 1], "Strictly decreasing"),
        ([100, 1, 100], [1, 1, 3], "High-low-high pattern"),
        ([50, 50, 50, 50], [1, 2, 3, 4], "All same prices"),
        ([1, 1000, 1], [1, 2, 1], "Extreme values"),
    ]
    
    for prices, expected, description in edge_cases:
        spanner = StockSpanner1()
        results = []
        
        for price in prices:
            results.append(spanner.next(price))
        
        status = "✓" if results == expected else "✗"
        print(f"{description:25} | {status} | prices: {prices} -> spans: {results}")


def analyze_amortized_complexity():
    """Analyze amortized time complexity"""
    print("\n=== Amortized Time Complexity Analysis ===")
    
    print("Why is the stack approach O(1) amortized?")
    print()
    print("Key insight: Each element is pushed and popped at most once")
    print()
    print("Example with prices [1, 2, 3, 4, 5]:")
    
    prices = [1, 2, 3, 4, 5]
    stack = []
    total_operations = 0
    
    for i, price in enumerate(prices):
        print(f"\nDay {i+1}: Price {price}")
        
        span = 1
        pops = 0
        
        while stack and stack[-1][0] <= price:
            stack.pop()
            pops += 1
            span += 1
        
        stack.append((price, span))
        
        operations = pops + 1  # pops + 1 push
        total_operations += operations
        
        print(f"  Operations: {pops} pops + 1 push = {operations}")
        print(f"  Stack size: {len(stack)}")
        print(f"  Total operations so far: {total_operations}")
    
    print(f"\nTotal operations: {total_operations}")
    print(f"Number of prices: {len(prices)}")
    print(f"Average operations per price: {total_operations / len(prices):.2f}")
    print("This demonstrates O(1) amortized complexity!")


def compare_stack_vs_brute_force():
    """Compare stack vs brute force approaches"""
    print("\n=== Stack vs Brute Force Comparison ===")
    
    test_cases = [
        [100, 80, 60, 70, 60, 75, 85],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [10, 10, 10, 10],
    ]
    
    for i, prices in enumerate(test_cases):
        print(f"\nTest case {i+1}: {prices}")
        
        # Stack approach
        spanner_stack = StockSpanner1()
        result_stack = []
        for price in prices:
            result_stack.append(spanner_stack.next(price))
        
        # Brute force approach
        spanner_brute = StockSpanner2()
        result_brute = []
        for price in prices:
            result_brute.append(spanner_brute.next(price))
        
        print(f"Stack approach:  {result_stack}")
        print(f"Brute force:     {result_brute}")
        print(f"Results match:   {'✓' if result_stack == result_brute else '✗'}")


def demonstrate_monotonic_stack_property():
    """Demonstrate monotonic stack property"""
    print("\n=== Monotonic Stack Property Demonstration ===")
    
    prices = [100, 80, 60, 70, 60, 75, 85]
    print(f"Processing prices: {prices}")
    print("Stack maintains decreasing order of prices:")
    
    stack = []
    
    for i, price in enumerate(prices):
        print(f"\nDay {i+1}: Processing price {price}")
        print(f"  Stack before: {[f'{p}(span:{s})' for p, s in stack]}")
        
        # Show what gets popped and why
        popped = []
        while stack and stack[-1][0] <= price:
            popped.append(stack.pop())
        
        if popped:
            print(f"  Popped: {[f'{p}(span:{s})' for p, s in popped]} (prices <= {price})")
        
        span = 1 + sum(s for _, s in popped)
        stack.append((price, span))
        
        print(f"  Stack after: {[f'{p}(span:{s})' for p, s in stack]}")
        
        # Verify monotonic property
        stack_prices = [p for p, s in stack]
        is_decreasing = all(stack_prices[j] > stack_prices[j+1] for j in range(len(stack_prices)-1))
        print(f"  Monotonic property: {'✓' if is_decreasing else '✗'} (prices: {stack_prices})")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Monotonic Stack", "O(1) amortized", "O(n)", "Each element pushed/popped once"),
        ("Brute Force", "O(n) per call", "O(n)", "Look back through all prices"),
        ("Stack with Indices", "O(1) amortized", "O(n)", "Similar to monotonic stack"),
        ("Previous Greater", "O(1) amortized", "O(n)", "Track previous greater elements"),
        ("Segment Tree", "O(log n) per call", "O(n)", "Range maximum queries"),
        ("Deque Approach", "O(1) amortized", "O(n)", "Similar to stack with deque"),
    ]
    
    print(f"{'Approach':<20} | {'Time per Call':<15} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<15} | {space_comp:<8} | {notes}")


if __name__ == "__main__":
    test_stock_spanner_implementations()
    demonstrate_stock_span_concept()
    demonstrate_stack_approach()
    visualize_stock_spans()
    demonstrate_monotonic_stack_property()
    analyze_amortized_complexity()
    test_edge_cases()
    compare_stack_vs_brute_force()
    analyze_time_complexity()
    benchmark_stock_spanner()

"""
Online Stock Span demonstrates monotonic stack applications for
streaming data processing, including amortized analysis and
multiple approaches for calculating consecutive day spans efficiently.
"""
