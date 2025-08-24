"""
1337. The K Weakest Rows in a Matrix - Multiple Approaches
Difficulty: Easy

You are given an m x n binary matrix mat of 1's (representing soldiers) and 0's (representing civilians). The soldiers are positioned in front of the civilians in each row.

A row i is weaker than a row j if one of the following is true:
- The number of soldiers in row i is less than the number of soldiers in row j.
- Both rows have the same number of soldiers and i < j.

Return the indices of the k weakest rows in the matrix ordered from weakest to strongest.
"""

from typing import List
import heapq

class KWeakestRowsInMatrix:
    """Multiple approaches to find k weakest rows"""
    
    def kWeakestRows_heap_approach(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 1: Min Heap (Optimal)
        
        Use min heap to track k weakest rows.
        
        Time: O(m log k), Space: O(k)
        """
        heap = []  # Max heap of (-soldiers, -row_idx) to keep k smallest
        
        for i, row in enumerate(mat):
            soldiers = sum(row)
            
            if len(heap) < k:
                # Negate for max heap behavior (we want min heap of k elements)
                heapq.heappush(heap, (-soldiers, -i))
            elif soldiers < -heap[0][0] or (soldiers == -heap[0][0] and i < -heap[0][1]):
                heapq.heappop(heap)
                heapq.heappush(heap, (-soldiers, -i))
        
        # Extract results and sort by strength (weakest first)
        result = []
        while heap:
            soldiers, row_idx = heapq.heappop(heap)
            result.append((-row_idx, -soldiers))
        
        # Sort by soldiers count, then by row index
        result.sort(key=lambda x: (x[1], x[0]))
        
        return [row_idx for row_idx, _ in result]
    
    def kWeakestRows_sorting(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 2: Sorting
        
        Sort all rows by strength and return first k.
        
        Time: O(m log m), Space: O(m)
        """
        rows_strength = []
        
        for i, row in enumerate(mat):
            soldiers = sum(row)
            rows_strength.append((soldiers, i))
        
        # Sort by soldiers count, then by row index
        rows_strength.sort()
        
        return [row_idx for _, row_idx in rows_strength[:k]]
    
    def kWeakestRows_binary_search(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 3: Binary Search + Sorting
        
        Use binary search to count soldiers efficiently.
        
        Time: O(m log n + m log m), Space: O(m)
        """
        def count_soldiers(row: List[int]) -> int:
            """Count soldiers using binary search"""
            left, right = 0, len(row)
            
            while left < right:
                mid = (left + right) // 2
                if row[mid] == 1:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        rows_strength = []
        
        for i, row in enumerate(mat):
            soldiers = count_soldiers(row)
            rows_strength.append((soldiers, i))
        
        rows_strength.sort()
        
        return [row_idx for _, row_idx in rows_strength[:k]]
    
    def kWeakestRows_priority_queue(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 4: Priority Queue with All Elements
        
        Add all rows to priority queue and extract k smallest.
        
        Time: O(m log m), Space: O(m)
        """
        heap = []
        
        for i, row in enumerate(mat):
            soldiers = sum(row)
            heapq.heappush(heap, (soldiers, i))
        
        result = []
        for _ in range(k):
            _, row_idx = heapq.heappop(heap)
            result.append(row_idx)
        
        return result
    
    def kWeakestRows_linear_scan(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 5: Linear Scan with Selection
        
        Use linear scan to find k weakest rows.
        
        Time: O(m * n + k * m), Space: O(m)
        """
        rows_strength = []
        
        for i, row in enumerate(mat):
            soldiers = sum(row)
            rows_strength.append((soldiers, i))
        
        # Use selection algorithm to find k smallest
        result = []
        used = [False] * len(rows_strength)
        
        for _ in range(k):
            min_idx = -1
            min_strength = float('inf')
            min_row_idx = float('inf')
            
            for i, (soldiers, row_idx) in enumerate(rows_strength):
                if not used[i]:
                    if (soldiers < min_strength or 
                        (soldiers == min_strength and row_idx < min_row_idx)):
                        min_strength = soldiers
                        min_row_idx = row_idx
                        min_idx = i
            
            if min_idx != -1:
                used[min_idx] = True
                result.append(min_row_idx)
        
        return result
    
    def kWeakestRows_bucket_sort(self, mat: List[List[int]], k: int) -> List[int]:
        """
        Approach 6: Bucket Sort
        
        Use bucket sort when number of soldiers is bounded.
        
        Time: O(m + n), Space: O(n + m)
        """
        n = len(mat[0]) if mat else 0
        buckets = [[] for _ in range(n + 1)]  # 0 to n soldiers
        
        # Distribute rows into buckets
        for i, row in enumerate(mat):
            soldiers = sum(row)
            buckets[soldiers].append(i)
        
        # Collect k weakest rows
        result = []
        for soldiers in range(n + 1):
            for row_idx in sorted(buckets[soldiers]):  # Sort by row index
                if len(result) < k:
                    result.append(row_idx)
                else:
                    return result
        
        return result


def test_k_weakest_rows():
    """Test k weakest rows algorithms"""
    solver = KWeakestRowsInMatrix()
    
    test_cases = [
        ([[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], 3, [2,0,3], "Example 1"),
        ([[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,0,0,0]], 2, [0,2], "Example 2"),
        ([[1,1,1],[1,1,1],[1,1,1]], 2, [0,1], "All same strength"),
        ([[0,0,0],[0,0,0],[0,0,0]], 2, [0,1], "All civilians"),
        ([[1]], 1, [0], "Single row"),
        ([[1,0],[0,1],[1,1]], 2, [1,0], "Mixed pattern"),
        ([[1,1,1,1],[0,0,0,0],[1,1,0,0]], 3, [1,2,0], "Different strengths"),
    ]
    
    algorithms = [
        ("Heap Approach", solver.kWeakestRows_heap_approach),
        ("Sorting", solver.kWeakestRows_sorting),
        ("Binary Search", solver.kWeakestRows_binary_search),
        ("Priority Queue", solver.kWeakestRows_priority_queue),
        ("Linear Scan", solver.kWeakestRows_linear_scan),
        ("Bucket Sort", solver.kWeakestRows_bucket_sort),
    ]
    
    print("=== Testing K Weakest Rows in Matrix ===")
    
    for mat, k, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print("Matrix:")
        for row in mat:
            print(f"  {row}")
        print(f"k: {k}")
        print(f"Expected: {expected}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(mat, k)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:20} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:40]}")


def demonstrate_heap_approach():
    """Demonstrate heap approach step by step"""
    print("\n=== Heap Approach Step-by-Step Demo ===")
    
    mat = [
        [1,1,0,0,0],
        [1,1,1,1,0],
        [1,0,0,0,0],
        [1,1,0,0,0],
        [1,1,1,1,1]
    ]
    k = 3
    
    print("Matrix:")
    for i, row in enumerate(mat):
        soldiers = sum(row)
        print(f"  Row {i}: {row} -> {soldiers} soldiers")
    
    print(f"\nFinding {k} weakest rows using max heap of size {k}:")
    
    heap = []
    
    for i, row in enumerate(mat):
        soldiers = sum(row)
        print(f"\nProcessing row {i} with {soldiers} soldiers:")
        
        if len(heap) < k:
            heapq.heappush(heap, (-soldiers, -i))
            print(f"  Added to heap: heap size = {len(heap)}")
        else:
            current_max_soldiers = -heap[0][0]
            current_max_row = -heap[0][1]
            
            print(f"  Current max in heap: {current_max_soldiers} soldiers (row {current_max_row})")
            
            if soldiers < current_max_soldiers or (soldiers == current_max_soldiers and i < current_max_row):
                heapq.heappop(heap)
                heapq.heappush(heap, (-soldiers, -i))
                print(f"  Replaced max with current row")
            else:
                print(f"  Current row is stronger, not added")
        
        heap_contents = [(-s, -r) for s, r in heap]
        print(f"  Heap contents: {heap_contents}")
    
    # Extract and sort results
    result = []
    while heap:
        soldiers, row_idx = heapq.heappop(heap)
        result.append((-row_idx, -soldiers))
    
    result.sort(key=lambda x: (x[1], x[0]))
    final_result = [row_idx for row_idx, _ in result]
    
    print(f"\nFinal result: {final_result}")


def visualize_row_strengths():
    """Visualize row strengths"""
    print("\n=== Row Strengths Visualization ===")
    
    mat = [
        [1,1,0,0,0],  # 2 soldiers
        [1,1,1,1,0],  # 4 soldiers
        [1,0,0,0,0],  # 1 soldier
        [1,1,0,0,0],  # 2 soldiers
        [1,1,1,1,1]   # 5 soldiers
    ]
    
    print("Matrix with soldier counts:")
    for i, row in enumerate(mat):
        soldiers = sum(row)
        visual = "".join("🪖" if x == 1 else "👤" for x in row)
        print(f"  Row {i}: {visual} ({soldiers} soldiers)")
    
    # Sort by strength
    rows_with_strength = [(sum(row), i, row) for i, row in enumerate(mat)]
    rows_with_strength.sort()
    
    print(f"\nRows sorted by strength (weakest to strongest):")
    for soldiers, row_idx, row in rows_with_strength:
        visual = "".join("🪖" if x == 1 else "👤" for x in row)
        print(f"  Row {row_idx}: {visual} ({soldiers} soldiers)")


def demonstrate_binary_search_optimization():
    """Demonstrate binary search optimization"""
    print("\n=== Binary Search Optimization ===")
    
    row = [1, 1, 1, 0, 0, 0, 0]
    print(f"Row: {row}")
    print("Since soldiers are at the front, we can use binary search")
    
    def count_soldiers_linear(row):
        """Linear counting"""
        return sum(row)
    
    def count_soldiers_binary(row):
        """Binary search counting"""
        left, right = 0, len(row)
        
        while left < right:
            mid = (left + right) // 2
            print(f"  Checking position {mid}: value = {row[mid]}")
            
            if row[mid] == 1:
                left = mid + 1
                print(f"    Found soldier, search right half: [{left}, {right})")
            else:
                right = mid
                print(f"    Found civilian, search left half: [{left}, {right})")
        
        return left
    
    print("\nLinear counting:")
    linear_result = count_soldiers_linear(row)
    print(f"  Result: {linear_result} soldiers")
    
    print(f"\nBinary search counting:")
    binary_result = count_soldiers_binary(row)
    print(f"  Result: {binary_result} soldiers")
    
    print(f"\nTime complexity:")
    print(f"  Linear: O(n) per row")
    print(f"  Binary search: O(log n) per row")


def benchmark_k_weakest_rows():
    """Benchmark different approaches"""
    import time
    import random
    
    algorithms = [
        ("Heap Approach", KWeakestRowsInMatrix().kWeakestRows_heap_approach),
        ("Sorting", KWeakestRowsInMatrix().kWeakestRows_sorting),
        ("Binary Search", KWeakestRowsInMatrix().kWeakestRows_binary_search),
        ("Priority Queue", KWeakestRowsInMatrix().kWeakestRows_priority_queue),
        ("Bucket Sort", KWeakestRowsInMatrix().kWeakestRows_bucket_sort),
    ]
    
    # Test parameters
    m, n = 1000, 50  # 1000 rows, 50 columns
    k = 10
    
    print(f"\n=== K Weakest Rows Performance Benchmark ===")
    print(f"Matrix size: {m}x{n}, k = {k}")
    
    # Generate test matrix
    mat = []
    for _ in range(m):
        soldiers = random.randint(0, n)
        row = [1] * soldiers + [0] * (n - soldiers)
        mat.append(row)
    
    for alg_name, alg_func in algorithms:
        start_time = time.time()
        
        try:
            result = alg_func(mat, k)
            end_time = time.time()
            print(f"{alg_name:20} | Time: {end_time - start_time:.4f}s | Result length: {len(result)}")
        except Exception as e:
            print(f"{alg_name:20} | ERROR: {str(e)[:30]}")


def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    solver = KWeakestRowsInMatrix()
    
    edge_cases = [
        ([[1]], 1, [0], "Single row, single column"),
        ([[0]], 1, [0], "Single civilian"),
        ([[1,0],[0,1]], 2, [1,0], "Two rows, different patterns"),
        ([[1,1],[1,1]], 1, [0], "Same strength, return first"),
        ([[0,0],[0,0]], 2, [0,1], "All civilians"),
        ([[1,1,1],[1,1,1]], 2, [0,1], "All soldiers, same strength"),
        ([[1,0,1]], 1, [0], "Non-contiguous soldiers (invalid but test)"),
    ]
    
    for mat, k, expected, description in edge_cases:
        try:
            result = solver.kWeakestRows_heap_approach(mat, k)
            status = "✓" if result == expected else "✗"
            print(f"{description:30} | {status} | mat: {mat}, k: {k} -> {result}")
        except Exception as e:
            print(f"{description:30} | ERROR: {str(e)[:30]}")


def compare_approaches():
    """Compare different approaches"""
    print("\n=== Approach Comparison ===")
    
    test_cases = [
        ([[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], 3),
        ([[1,0,0,0],[1,1,1,1],[1,0,0,0],[1,0,0,0]], 2),
        ([[1,1,1],[1,1,1],[1,1,1]], 2),
    ]
    
    solver = KWeakestRowsInMatrix()
    
    approaches = [
        ("Heap", solver.kWeakestRows_heap_approach),
        ("Sorting", solver.kWeakestRows_sorting),
        ("Binary Search", solver.kWeakestRows_binary_search),
        ("Priority Queue", solver.kWeakestRows_priority_queue),
        ("Bucket Sort", solver.kWeakestRows_bucket_sort),
    ]
    
    for i, (mat, k) in enumerate(test_cases):
        print(f"\nTest case {i+1}: k={k}")
        for row in mat:
            print(f"  {row}")
        
        results = {}
        
        for name, func in approaches:
            try:
                result = func(mat, k)
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
        ("Heap Approach", "O(m log k)", "O(k)", "Min heap of size k"),
        ("Sorting", "O(m log m)", "O(m)", "Sort all rows by strength"),
        ("Binary Search", "O(m log n + m log m)", "O(m)", "Binary search + sorting"),
        ("Priority Queue", "O(m log m)", "O(m)", "Heap with all elements"),
        ("Linear Scan", "O(k * m)", "O(m)", "Selection algorithm"),
        ("Bucket Sort", "O(m + n)", "O(n + m)", "When soldiers count is bounded"),
    ]
    
    print(f"{'Approach':<20} | {'Time':<20} | {'Space':<8} | {'Notes'}")
    print("-" * 75)
    
    for approach, time_comp, space_comp, notes in approaches:
        print(f"{approach:<20} | {time_comp:<20} | {space_comp:<8} | {notes}")
    
    print(f"\nWhere m = number of rows, n = number of columns, k = result size")


def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    # Application 1: Military unit assessment
    print("1. Military Unit Strength Assessment:")
    units = [
        [1,1,1,0,0],  # Unit A: 3 active soldiers
        [1,1,1,1,1],  # Unit B: 5 active soldiers  
        [1,0,0,0,0],  # Unit C: 1 active soldier
        [1,1,0,0,0],  # Unit D: 2 active soldiers
    ]
    
    solver = KWeakestRowsInMatrix()
    weakest_units = solver.kWeakestRows_heap_approach(units, 2)
    
    print("  Unit strengths:")
    for i, unit in enumerate(units):
        active = sum(unit)
        status = "🟢" * active + "🔴" * (len(unit) - active)
        print(f"    Unit {chr(65+i)}: {status} ({active} active)")
    
    print(f"  Two weakest units for reinforcement: {[chr(65+i) for i in weakest_units]}")
    
    # Application 2: Server load balancing
    print("\n2. Server Load Balancing:")
    servers = [
        [1,1,0,0,0,0],  # Server 1: 2 busy cores
        [1,1,1,1,0,0],  # Server 2: 4 busy cores
        [1,0,0,0,0,0],  # Server 3: 1 busy core
        [1,1,1,0,0,0],  # Server 4: 3 busy cores
    ]
    
    least_loaded = solver.kWeakestRows_heap_approach(servers, 2)
    
    print("  Server loads:")
    for i, server in enumerate(servers):
        busy = sum(server)
        load_bar = "█" * busy + "░" * (len(server) - busy)
        print(f"    Server {i+1}: {load_bar} ({busy}/{len(server)} cores busy)")
    
    print(f"  Best servers for new tasks: {[f'Server {i+1}' for i in least_loaded]}")


if __name__ == "__main__":
    test_k_weakest_rows()
    demonstrate_heap_approach()
    visualize_row_strengths()
    demonstrate_binary_search_optimization()
    demonstrate_real_world_applications()
    test_edge_cases()
    compare_approaches()
    analyze_time_complexity()
    benchmark_k_weakest_rows()

"""
The K Weakest Rows in a Matrix demonstrates heap applications for
selection problems, including binary search optimization and multiple
approaches for finding top-k elements with custom comparison criteria.
"""
