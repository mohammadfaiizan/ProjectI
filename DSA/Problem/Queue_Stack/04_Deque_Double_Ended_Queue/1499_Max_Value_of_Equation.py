"""
1499. Max Value of Equation - Multiple Approaches
Difficulty: Hard

You are given an array points containing the coordinates of points on a 2D plane, sorted by the x-coordinates, where points[i] = [xi, yi]. You are also given an integer k.

Return the maximum value of the equation yi + yj + |xi - xj| where |xi - xj| <= k and i < j.

It is guaranteed that there exists at least one pair of points that satisfy the constraint |xi - xj| <= k.
"""

from typing import List, Deque
from collections import deque
import heapq

class MaxValueOfEquation:
    """Multiple approaches to find max value of equation"""
    
    def findMaxValueOfEquation_deque_approach(self, points: List[List[int]], k: int) -> int:
        """
        Approach 1: Monotonic Deque (Optimal)
        
        Transform equation and use deque to maintain maximum values.
        
        Time: O(n), Space: O(n)
        """
        # Transform: yi + yj + |xi - xj| where xi <= xj (sorted)
        # = yi + yj + xj - xi = (yi - xi) + (yj + xj)
        # For each j, find maximum (yi - xi) where xi >= xj - k
        
        dq: Deque[int] = deque()  # Store indices
        max_value = float('-inf')
        
        for j in range(len(points)):
            xj, yj = points[j]
            
            # Remove points where xi < xj - k (outside valid range)
            while dq and points[dq[0]][0] < xj - k:
                dq.popleft()
            
            # Calculate max value using best previous point
            if dq:
                i = dq[0]
                xi, yi = points[i]
                value = yi + yj + xj - xi
                max_value = max(max_value, value)
            
            # Maintain decreasing order of (yi - xi)
            while dq and (points[dq[-1]][1] - points[dq[-1]][0]) <= (yj - xj):
                dq.pop()
            
            dq.append(j)
        
        return max_value
    
    def findMaxValueOfEquation_heap_approach(self, points: List[List[int]], k: int) -> int:
        """
        Approach 2: Max Heap with Lazy Deletion
        
        Use max heap to track maximum (yi - xi) values.
        
        Time: O(n log n), Space: O(n)
        """
        # Max heap to store (-(yi - xi), xi, i)
        heap = []
        max_value = float('-inf')
        
        for j in range(len(points)):
            xj, yj = points[j]
            
            # Remove points outside valid range (lazy deletion)
            while heap and heap[0][1] < xj - k:
                heapq.heappop(heap)
            
            # Calculate max value using best previous point
            if heap:
                neg_diff, xi, i = heap[0]
                yi_minus_xi = -neg_diff
                value = yi_minus_xi + yj + xj
                max_value = max(max_value, value)
            
            # Add current point to heap
            heapq.heappush(heap, (-(yj - xj), xj, j))
        
        return max_value
    
    def findMaxValueOfEquation_brute_force(self, points: List[List[int]], k: int) -> int:
        """
        Approach 3: Brute Force
        
        Check all valid pairs of points.
        
        Time: O(n²), Space: O(1)
        """
        n = len(points)
        max_value = float('-inf')
        
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                
                if abs(xi - xj) <= k:
                    value = yi + yj + abs(xi - xj)
                    max_value = max(max_value, value)
                else:
                    break  # Since points are sorted, no need to check further
        
        return max_value
    
    def findMaxValueOfEquation_sliding_window(self, points: List[List[int]], k: int) -> int:
        """
        Approach 4: Sliding Window with Deque
        
        Use sliding window to maintain valid range.
        
        Time: O(n), Space: O(n)
        """
        dq = deque()  # Store (yi - xi, i)
        max_value = float('-inf')
        
        for j in range(len(points)):
            xj, yj = points[j]
            
            # Remove points outside sliding window
            while dq and points[dq[0][1]][0] < xj - k:
                dq.popleft()
            
            # Calculate value with best point in window
            if dq:
                yi_minus_xi, i = dq[0]
                value = yi_minus_xi + yj + xj
                max_value = max(max_value, value)
            
            # Maintain decreasing order of yi - xi
            current_diff = yj - xj
            while dq and dq[-1][0] <= current_diff:
                dq.pop()
            
            dq.append((current_diff, j))
        
        return max_value
    
    def findMaxValueOfEquation_segment_tree(self, points: List[List[int]], k: int) -> int:
        """
        Approach 5: Segment Tree for Range Maximum Query
        
        Use segment tree to find maximum yi - xi in range.
        
        Time: O(n log n), Space: O(n)
        """
        n = len(points)
        
        # Coordinate compression for x-coordinates
        x_coords = sorted(set(point[0] for point in points))
        coord_map = {x: i for i, x in enumerate(x_coords)}
        
        # Build segment tree
        tree_size = len(x_coords)
        tree = [float('-inf')] * (4 * tree_size)
        
        def update(node: int, start: int, end: int, idx: int, val: int) -> None:
            if start == end:
                tree[node] = max(tree[node], val)
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    update(2 * node, start, mid, idx, val)
                else:
                    update(2 * node + 1, mid + 1, end, idx, val)
                tree[node] = max(tree[2 * node], tree[2 * node + 1])
        
        def query_max(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return float('-inf')
            if l <= start and end <= r:
                return tree[node]
            
            mid = (start + end) // 2
            left_max = query_max(2 * node, start, mid, l, r)
            right_max = query_max(2 * node + 1, mid + 1, end, l, r)
            return max(left_max, right_max)
        
        max_value = float('-inf')
        
        for j in range(n):
            xj, yj = points[j]
            
            # Find range of valid x-coordinates [xj - k, xj - 1]
            left_x = xj - k
            right_x = xj - 1
            
            # Binary search for range in compressed coordinates
            left_idx = 0
            right_idx = len(x_coords) - 1
            
            # Find leftmost valid coordinate
            while left_idx <= right_idx:
                mid = (left_idx + right_idx) // 2
                if x_coords[mid] >= left_x:
                    right_idx = mid - 1
                else:
                    left_idx = mid + 1
            
            left_bound = left_idx
            
            # Find rightmost valid coordinate
            left_idx = 0
            right_idx = len(x_coords) - 1
            
            while left_idx <= right_idx:
                mid = (left_idx + right_idx) // 2
                if x_coords[mid] <= right_x:
                    left_idx = mid + 1
                else:
                    right_idx = mid - 1
            
            right_bound = right_idx
            
            # Query maximum in valid range
            if left_bound <= right_bound:
                max_yi_minus_xi = query_max(1, 0, tree_size - 1, left_bound, right_bound)
                if max_yi_minus_xi != float('-inf'):
                    value = max_yi_minus_xi + yj + xj
                    max_value = max(max_value, value)
            
            # Update segment tree with current point
            update(1, 0, tree_size - 1, coord_map[xj], yj - xj)
        
        return max_value
    
    def findMaxValueOfEquation_two_pointers(self, points: List[List[int]], k: int) -> int:
        """
        Approach 6: Two Pointers with Optimization
        
        Use two pointers to maintain valid window.
        
        Time: O(n²), Space: O(1)
        """
        n = len(points)
        max_value = float('-inf')
        
        for j in range(1, n):
            xj, yj = points[j]
            
            # Find all valid i where xi >= xj - k
            for i in range(j):
                xi, yi = points[i]
                
                if xi >= xj - k:
                    value = yi + yj + xj - xi
                    max_value = max(max_value, value)
        
        return max_value
    
    def findMaxValueOfEquation_optimized_scan(self, points: List[List[int]], k: int) -> int:
        """
        Approach 7: Optimized Linear Scan
        
        Maintain maximum yi - xi in sliding window.
        
        Time: O(n), Space: O(1)
        """
        max_value = float('-inf')
        max_yi_minus_xi = float('-inf')
        left = 0
        
        for right in range(1, len(points)):
            xr, yr = points[right]
            
            # Move left pointer to maintain valid window
            while left < right and points[left][0] < xr - k:
                left += 1
            
            # Update max_yi_minus_xi for current window
            max_yi_minus_xi = float('-inf')
            for i in range(left, right):
                xi, yi = points[i]
                max_yi_minus_xi = max(max_yi_minus_xi, yi - xi)
            
            # Calculate max value
            if max_yi_minus_xi != float('-inf'):
                value = max_yi_minus_xi + yr + xr
                max_value = max(max_value, value)
        
        return max_value


def test_max_value_of_equation():
    """Test max value of equation algorithms"""
    solver = MaxValueOfEquation()
    
    test_cases = [
        ([[1,3],[2,0],[5,10],[6,-10]], 1, 4, "Example 1"),
        ([[0,0],[3,0],[9,2]], 3, 3, "Example 2"),
        ([[-17,5],[-10,-8],[1,8],[3,-2]], 6, 20, "Negative coordinates"),
        ([[1,1],[2,2]], 1, 5, "Two points"),
        ([[0,0],[1,1],[2,2]], 2, 4, "Three points"),
        ([[-1,-1],[0,0],[1,1]], 2, 2, "Negative start"),
        ([[1,10],[2,1],[3,15]], 1, 26, "Large y values"),
        ([[0,0],[5,5],[10,10]], 5, 15, "Linear pattern"),
        ([[-5,1],[0,2],[5,3]], 10, 13, "Wide range"),
        ([[1,1],[1,2],[1,3]], 0, 5, "Same x-coordinate"),
    ]
    
    algorithms = [
        ("Deque Approach", solver.findMaxValueOfEquation_deque_approach),
        ("Heap Approach", solver.findMaxValueOfEquation_heap_approach),
        ("Brute Force", solver.findMaxValueOfEquation_brute_force),
        ("Sliding Window", solver.findMaxValueOfEquation_sliding_window),
        ("Segment Tree", solver.findMaxValueOfEquation_segment_tree),
        ("Two Pointers", solver.findMaxValueOfEquation_two_pointers),
        ("Optimized Scan", solver.findMaxValueOfEquation_optimized_scan),
    ]
    
    print("=== Testing Max Value of Equation ===")
    
    for points, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Points: {points}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(points, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_equation_transformation():
    """Demonstrate equation transformation"""
    print("\n=== Equation Transformation Demonstration ===")
    
    points = [[1, 3], [2, 0], [5, 10], [6, -10]]
    k = 1
    
    print(f"Points: {points}")
    print(f"k: {k}")
    
    print("\nOriginal equation: yi + yj + |xi - xj| where |xi - xj| <= k and i < j")
    print("Since points are sorted by x-coordinate: xi <= xj, so |xi - xj| = xj - xi")
    print("Transformed equation: yi + yj + xj - xi = (yi - xi) + (yj + xj)")
    
    print("\nFor each point j, we want to find the maximum (yi - xi) among valid previous points i")
    print("Valid means: xi >= xj - k (equivalent to |xi - xj| <= k)")
    
    print(f"\nCalculating yi - xi for each point:")
    for i, (x, y) in enumerate(points):
        diff = y - x
        print(f"  Point {i}: ({x}, {y}) -> yi - xi = {y} - {x} = {diff}")
    
    print(f"\nFinding maximum equation value:")
    max_value = float('-inf')
    
    for j in range(1, len(points)):
        xj, yj = points[j]
        print(f"\nFor point j={j}: ({xj}, {yj})")
        print(f"  Need xi >= {xj} - {k} = {xj - k}")
        
        valid_points = []
        for i in range(j):
            xi, yi = points[i]
            if xi >= xj - k:
                yi_minus_xi = yi - xi
                value = yi_minus_xi + yj + xj
                valid_points.append((i, xi, yi, yi_minus_xi, value))
        
        if valid_points:
            for i, xi, yi, diff, value in valid_points:
                print(f"    Point i={i}: ({xi}, {yi}) -> (yi - xi) + (yj + xj) = {diff} + ({yj} + {xj}) = {value}")
            
            best_value = max(valid_points, key=lambda x: x[4])[4]
            max_value = max(max_value, best_value)
            print(f"  Best value for j={j}: {best_value}")
    
    print(f"\nMaximum equation value: {max_value}")


def demonstrate_deque_approach():
    """Demonstrate deque approach step by step"""
    print("\n=== Deque Approach Step-by-Step Demo ===")
    
    points = [[1, 3], [2, 0], [5, 10], [6, -10]]
    k = 1
    
    print(f"Points: {points}")
    print(f"k: {k}")
    print("Deque maintains points in decreasing order of (yi - xi)")
    
    dq = deque()
    max_value = float('-inf')
    
    for j in range(len(points)):
        xj, yj = points[j]
        print(f"\nStep {j+1}: Processing point ({xj}, {yj})")
        
        # Remove points outside valid range
        removed_outside = []
        while dq and points[dq[0]][0] < xj - k:
            removed = dq.popleft()
            removed_outside.append(removed)
        
        if removed_outside:
            print(f"  Removed outside range: indices {removed_outside}")
        
        # Show current valid points
        if dq:
            valid_indices = list(dq)
            valid_info = []
            for idx in valid_indices:
                xi, yi = points[idx]
                diff = yi - xi
                valid_info.append(f"idx {idx}: ({xi},{yi}) diff={diff}")
            print(f"  Valid points: {valid_info}")
            
            # Calculate value with best point
            best_idx = dq[0]
            xi, yi = points[best_idx]
            value = (yi - xi) + (yj + xj)
            max_value = max(max_value, value)
            print(f"  Best previous point: idx {best_idx}")
            print(f"  Equation value: ({yi} - {xi}) + ({yj} + {xj}) = {value}")
            print(f"  Max value so far: {max_value}")
        else:
            print(f"  No valid previous points")
        
        # Maintain decreasing order
        current_diff = yj - xj
        removed_smaller = []
        while dq and (points[dq[-1]][1] - points[dq[-1]][0]) <= current_diff:
            removed = dq.pop()
            removed_smaller.append(f"idx {removed} (diff={points[removed][1] - points[removed][0]})")
        
        if removed_smaller:
            print(f"  Removed smaller/equal diffs: {removed_smaller}")
        
        dq.append(j)
        print(f"  Added current point to deque")
        print(f"  Deque after: {list(dq)}")
    
    print(f"\nFinal maximum value: {max_value}")


def visualize_geometric_interpretation():
    """Visualize geometric interpretation"""
    print("\n=== Geometric Interpretation ===")
    
    points = [[1, 3], [2, 0], [5, 10], [6, -10]]
    k = 1
    
    print(f"Points: {points}")
    print(f"Constraint: |xi - xj| <= {k}")
    
    print("\nGeometric meaning:")
    print("- We have points on a 2D plane")
    print("- For each pair (i, j) where i < j, we calculate yi + yj + |xi - xj|")
    print("- This represents the sum of y-coordinates plus the Manhattan distance in x")
    print("- We want the maximum such value among all valid pairs")
    
    print(f"\nValid pairs (|xi - xj| <= {k}):")
    
    n = len(points)
    valid_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            
            if abs(xi - xj) <= k:
                value = yi + yj + abs(xi - xj)
                valid_pairs.append((i, j, xi, yi, xj, yj, value))
                print(f"  Pair ({i},{j}): ({xi},{yi}) and ({xj},{yj})")
                print(f"    Distance: |{xi} - {xj}| = {abs(xi - xj)} <= {k} ✓")
                print(f"    Value: {yi} + {yj} + {abs(xi - xj)} = {value}")
    
    if valid_pairs:
        best_pair = max(valid_pairs, key=lambda x: x[6])
        i, j, xi, yi, xj, yj, value = best_pair
        print(f"\nBest pair: ({i},{j}) with value {value}")
        print(f"  Points: ({xi},{yi}) and ({xj},{yj})")


def demonstrate_sliding_window_concept():
    """Demonstrate sliding window concept"""
    print("\n=== Sliding Window Concept ===")
    
    points = [[0, 1], [1, 3], [2, 0], [4, 2], [5, 4]]
    k = 2
    
    print(f"Points: {points}")
    print(f"k: {k}")
    
    print("\nSliding window approach:")
    print("For each point j, maintain a window of previous points i where xi >= xj - k")
    
    for j in range(1, len(points)):
        xj, yj = points[j]
        min_x = xj - k
        
        print(f"\nPoint j={j}: ({xj}, {yj})")
        print(f"  Window constraint: xi >= {xj} - {k} = {min_x}")
        
        valid_window = []
        for i in range(j):
            xi, yi = points[i]
            if xi >= min_x:
                diff = yi - xi
                valid_window.append((i, xi, yi, diff))
        
        if valid_window:
            print(f"  Valid window points:")
            for i, xi, yi, diff in valid_window:
                print(f"    Point {i}: ({xi}, {yi}) -> yi - xi = {diff}")
            
            best_diff = max(valid_window, key=lambda x: x[3])[3]
            value = best_diff + yj + xj
            print(f"  Best yi - xi in window: {best_diff}")
            print(f"  Equation value: {best_diff} + {yj} + {xj} = {value}")
        else:
            print(f"  No valid points in window")


def benchmark_max_value_equation():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Deque Approach", MaxValueOfEquation().findMaxValueOfEquation_deque_approach),
        ("Heap Approach", MaxValueOfEquation().findMaxValueOfEquation_heap_approach),
        ("Brute Force", MaxValueOfEquation().findMaxValueOfEquation_brute_force),
        ("Sliding Window", MaxValueOfEquation().findMaxValueOfEquation_sliding_window),
        ("Two Pointers", MaxValueOfEquation().findMaxValueOfEquation_two_pointers),
    ]
    
    # Test with different array sizes
    test_sizes = [(100, 10), (1000, 50), (5000, 100)]
    
    print("\n=== Max Value of Equation Performance Benchmark ===")
    
    for size, k in test_sizes:
        print(f"\n--- Array Size: {size}, k: {k} ---")
        
        # Generate random sorted points
        points = []
        for i in range(size):
            x = i + random.randint(0, 2)  # Ensure sorted with some randomness
            y = random.randint(-100, 100)
            points.append([x, y])
        
        for alg_name, alg_func in algorithms:
            start_time = time.time()
            
            try:
                result = alg_func(points, k)
                end_time = time.time()
                print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = MaxValueOfEquation()
    
    edge_cases = [
        ([[0,0],[1,1]], 1, 2, "Two points"),
        ([[0,0],[1,1],[2,2]], 1, 3, "Three points, k=1"),
        ([[0,0],[1,1],[2,2]], 2, 4, "Three points, k=2"),
        ([[-1,-1],[0,0],[1,1]], 2, 2, "Negative coordinates"),
        ([[0,10],[1,0]], 1, 11, "Large y difference"),
        ([[0,0],[10,0]], 5, 0, "k smaller than distance"),
        ([[1,1],[1,2],[1,3]], 0, 5, "Same x-coordinates"),
        ([[0,0],[1,10],[2,0]], 2, 12, "Peak in middle"),
        ([[-10,5],[0,0],[10,5]], 20, 25, "Wide range"),
        ([[0,1],[1,0],[2,1]], 1, 3, "Alternating pattern"),
    ]
    
    for points, k, expected, description in edge_cases:
        try:
            result = solver.findMaxValueOfEquation_deque_approach(points, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:25} | {status} | points: {points}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:25} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([[1,3],[2,0],[5,10],[6,-10]], 1),
        ([[0,0],[3,0],[9,2]], 3),
        ([[-17,5],[-10,-8],[1,8],[3,-2]], 6),
        ([[1,1],[2,2],[3,3]], 2),
    ]
    
    solver = MaxValueOfEquation()
    
    approaches = [
        ("Deque", solver.findMaxValueOfEquation_deque_approach),
        ("Heap", solver.findMaxValueOfEquation_heap_approach),
        ("Brute Force", solver.findMaxValueOfEquation_brute_force),
        ("Sliding Window", solver.findMaxValueOfEquation_sliding_window),
        ("Two Pointers", solver.findMaxValueOfEquation_two_pointers),
    ]
    
    for i, (points, k) in enumerate(test_cases):
        print(f"\nTest case {i+1}: points={points}, k={k}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(points, k)
                results[name] = result
                print(f"{name:15} | Result: {result}")
            except Exception as e:
                print(f"{name:15} | ERROR: {str(e)[:40]}")
        
        # Check consistency
        if results:
            first_result = list(results.values())[0]
            all_same = all(result == first_result for result in results.values())
            print(f"All approaches agree: {'✓' if all_same else '✗'}")


def analyze_time_complexity():
    """Analyze time complexity of different approaches"""
    print("\n=== Time Complexity Analysis ===")
    
    approaches = [
        ("Deque Approach", "O(n)", "O(n)", "Each point added/removed once"),
        ("Heap Approach", "O(n log n)", "O(n)", "Heap operations with lazy deletion"),
        ("Brute Force", "O(n²)", "O(1)", "Check all pairs of points"),
        ("Sliding Window", "O(n)", "O(n)", "Deque-based sliding window"),
        ("Segment Tree", "O(n log n)", "O(n)", "Range maximum queries"),
        ("Two Pointers", "O(n²)", "O(1)", "Two pointers with nested loop"),
        ("Optimized Scan", "O(n²)", "O(1)", "Linear scan with window update"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<12} | {'Space':<8} | {'Notes'}")
    print("-" * 70)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<12} | {space_comp:<8} | {notes}")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Sensor network optimization
    print("1. Sensor Network - Maximum signal strength between sensor pairs:")
    sensors = [[0, 50], [2, 30], [5, 80], [7, 20]]  # [position, signal_strength]
    max_distance = 3  # Maximum communication range
    
    solver = MaxValueOfEquation()
    max_signal = solver.findMaxValueOfEquation_deque_approach(sensors, max_distance)
    
    print(f"  Sensor positions and strengths: {sensors}")
    print(f"  Maximum communication range: {max_distance}")
    print(f"  Maximum combined signal strength: {max_signal}")
    print("  (Equation: strength_i + strength_j + distance)")
    
    # Application 2: Delivery route optimization
    print("\n2. Delivery Route - Maximum profit between delivery points:")
    deliveries = [[1, 100], [3, 80], [6, 120], [8, 60]]  # [time, profit]
    time_constraint = 4  # Maximum time difference between deliveries
    
    max_profit = solver.findMaxValueOfEquation_deque_approach(deliveries, time_constraint)
    
    print(f"  Delivery times and profits: {deliveries}")
    print(f"  Time constraint: {time_constraint} units")
    print(f"  Maximum combined profit: {max_profit}")
    print("  (Includes travel time cost in equation)")
    
    # Application 3: Investment timing
    print("\n3. Investment Timing - Maximum return with timing constraints:")
    investments = [[0, 200], [2, 150], [5, 300], [7, 100]]  # [month, return]
    timing_limit = 3  # Maximum months between investments
    
    max_return = solver.findMaxValueOfEquation_deque_approach(investments, timing_limit)
    
    print(f"  Investment months and returns: {investments}")
    print(f"  Timing constraint: {timing_limit} months")
    print(f"  Maximum combined return: {max_return}")
    print("  (Equation accounts for timing penalty)")


def demonstrate_monotonic_deque_property():
    """Demonstrate monotonic deque property"""
    print("\n=== Monotonic Deque Property ===")
    
    points = [[1, 5], [2, 1], [3, 8], [4, 2], [5, 6]]
    k = 2
    
    print(f"Points: {points}")
    print("Deque maintains decreasing order of (yi - xi) values")
    
    print(f"\nCalculating yi - xi for each point:")
    for i, (x, y) in enumerate(points):
        diff = y - x
        print(f"  Point {i}: ({x}, {y}) -> yi - xi = {y} - {x} = {diff}")
    
    print(f"\nProcessing with deque:")
    dq = deque()
    
    for j, (xj, yj) in enumerate(points):
        print(f"\nStep {j+1}: Processing point ({xj}, {yj})")
        current_diff = yj - xj
        
        # Show what gets removed and why
        removed = []
        while dq and (points[dq[-1]][1] - points[dq[-1]][0]) <= current_diff:
            removed_idx = dq.pop()
            removed_diff = points[removed_idx][1] - points[removed_idx][0]
            removed.append(f"idx {removed_idx} (diff={removed_diff})")
        
        if removed:
            print(f"  Removed: {removed} (smaller/equal than {current_diff})")
        
        dq.append(j)
        
        # Show current deque state
        deque_info = []
        for idx in dq:
            x, y = points[idx]
            diff = y - x
            deque_info.append(f"idx {idx} (diff={diff})")
        
        print(f"  Deque: {deque_info}")
        
        # Verify monotonic property
        diffs = [points[idx][1] - points[idx][0] for idx in dq]
        is_decreasing = all(diffs[i] >= diffs[i+1] for i in range(len(diffs)-1))
        print(f"  Monotonic (decreasing): {'✓' if is_decreasing else '✗'} (diffs: {diffs})")


if __name__ == "__main__":
    test_max_value_of_equation()
    demonstrate_equation_transformation()
    demonstrate_deque_approach()
    visualize_geometric_interpretation()
    demonstrate_sliding_window_concept()
    demonstrate_monotonic_deque_property()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_max_value_equation()

"""
Max Value of Equation demonstrates advanced deque applications for geometric
optimization problems, including equation transformation techniques and
multiple approaches for finding optimal point pairs with distance constraints.
"""
