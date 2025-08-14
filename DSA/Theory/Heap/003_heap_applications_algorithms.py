"""
Heap Applications and Advanced Algorithms
=========================================

Topics: Advanced heap algorithms, specialized applications, optimization problems
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber
Difficulty: Medium to Hard
Time Complexity: Varies by application
Space Complexity: O(n) for heap storage + problem-specific
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import heapq
from collections import defaultdict, Counter
import math

class HeapApplications:
    
    def __init__(self):
        """Initialize with solution tracking"""
        self.solution_steps = []
        self.operation_count = 0
    
    # ==========================================
    # 1. K-ELEMENT PROBLEMS
    # ==========================================
    
    def k_largest_elements(self, nums: List[int], k: int) -> List[int]:
        """
        Find k largest elements using min heap
        
        Company: Amazon, Google, Microsoft
        Difficulty: Medium
        Time: O(n log k), Space: O(k)
        
        Algorithm: Maintain min heap of size k
        """
        if not nums or k <= 0:
            return []
        
        # Use min heap to keep k largest elements
        min_heap = []
        
        print(f"Finding {k} largest elements from: {nums}")
        print("Using min heap of size k for optimal space complexity")
        print()
        
        for i, num in enumerate(nums):
            print(f"Step {i+1}: Processing {num}")
            
            if len(min_heap) < k:
                # Heap not full, just add
                heapq.heappush(min_heap, num)
                print(f"   Heap size < k, added {num}")
                print(f"   Current heap: {min_heap}")
            else:
                # Heap full, compare with minimum
                if num > min_heap[0]:
                    # Replace minimum with current number
                    replaced = heapq.heapreplace(min_heap, num)
                    print(f"   {num} > min {replaced}, replaced minimum")
                    print(f"   Current heap: {min_heap}")
                else:
                    print(f"   {num} ≤ min {min_heap[0]}, not added")
            
            # Show current k largest
            current_k_largest = sorted(min_heap, reverse=True)
            print(f"   Current {len(min_heap)} largest: {current_k_largest}")
            print()
        
        # Convert min heap to sorted list (largest first)
        result = sorted(min_heap, reverse=True)
        print(f"Final {k} largest elements: {result}")
        return result
    
    def k_smallest_elements(self, nums: List[int], k: int) -> List[int]:
        """
        Find k smallest elements using max heap
        
        Time: O(n log k), Space: O(k)
        """
        if not nums or k <= 0:
            return []
        
        # Use max heap (negate values) to keep k smallest elements
        max_heap = []
        
        print(f"Finding {k} smallest elements from: {nums}")
        print("Using max heap of size k")
        print()
        
        for i, num in enumerate(nums):
            print(f"Step {i+1}: Processing {num}")
            
            if len(max_heap) < k:
                # Heap not full, add negated value for max heap behavior
                heapq.heappush(max_heap, -num)
                print(f"   Added {num} (as {-num}) to heap")
                print(f"   Current heap: {[-x for x in max_heap]}")
            else:
                # Heap full, compare with maximum
                if num < -max_heap[0]:
                    # Replace maximum with current number
                    replaced = -heapq.heapreplace(max_heap, -num)
                    print(f"   {num} < max {replaced}, replaced maximum")
                    print(f"   Current heap: {[-x for x in max_heap]}")
                else:
                    print(f"   {num} ≥ max {-max_heap[0]}, not added")
            
            # Show current k smallest
            current_k_smallest = sorted([-x for x in max_heap])
            print(f"   Current {len(max_heap)} smallest: {current_k_smallest}")
            print()
        
        # Convert max heap to sorted list
        result = sorted([-x for x in max_heap])
        print(f"Final {k} smallest elements: {result}")
        return result
    
    def kth_largest_element(self, nums: List[int], k: int) -> int:
        """
        Find kth largest element using min heap
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n log k), Space: O(k)
        """
        print(f"Finding {k}th largest element from: {nums}")
        
        # Maintain min heap of size k
        min_heap = []
        
        for num in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, num)
            elif num > min_heap[0]:
                heapq.heapreplace(min_heap, num)
        
        kth_largest = min_heap[0]
        print(f"The {k}th largest element is: {kth_largest}")
        print(f"Heap containing {k} largest: {sorted(min_heap, reverse=True)}")
        
        return kth_largest
    
    def top_k_frequent_elements(self, nums: List[int], k: int) -> List[int]:
        """
        Find k most frequent elements using heap
        
        Company: Amazon, Facebook, Google
        Difficulty: Medium
        Time: O(n log k), Space: O(n)
        """
        # Count frequencies
        freq_map = Counter(nums)
        
        print(f"Finding {k} most frequent elements from: {nums}")
        print(f"Frequency map: {dict(freq_map)}")
        print()
        
        # Use min heap to keep k most frequent elements
        # Store (frequency, element) pairs
        min_heap = []
        
        for element, frequency in freq_map.items():
            print(f"Processing element {element} with frequency {frequency}")
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (frequency, element))
                print(f"   Added to heap: {min_heap}")
            elif frequency > min_heap[0][0]:
                replaced = heapq.heapreplace(min_heap, (frequency, element))
                print(f"   Replaced {replaced} with ({frequency}, {element})")
                print(f"   Current heap: {min_heap}")
            else:
                print(f"   Frequency {frequency} ≤ min frequency {min_heap[0][0]}, not added")
            
            print()
        
        # Extract elements from heap
        result = [element for freq, element in min_heap]
        print(f"Top {k} frequent elements: {result}")
        return result
    
    # ==========================================
    # 2. MERGE ALGORITHMS
    # ==========================================
    
    def merge_k_sorted_lists(self, lists: List[List[int]]) -> List[int]:
        """
        Merge k sorted lists using min heap
        
        Company: Google, Facebook, Amazon
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        where n is total elements, k is number of lists
        """
        if not lists:
            return []
        
        # Min heap to store (value, list_index, element_index)
        min_heap = []
        result = []
        
        print(f"Merging {len(lists)} sorted lists:")
        for i, lst in enumerate(lists):
            print(f"   List {i}: {lst}")
        print()
        
        # Initialize heap with first element from each non-empty list
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(min_heap, (lst[0], i, 0))
                print(f"Initial: Added {lst[0]} from list {i}")
        
        print(f"Initial heap: {min_heap}")
        print()
        
        step = 1
        while min_heap:
            print(f"Step {step}:")
            
            # Extract minimum element
            value, list_idx, elem_idx = heapq.heappop(min_heap)
            result.append(value)
            
            print(f"   Extracted minimum: {value} from list {list_idx}")
            print(f"   Current result: {result}")
            
            # Add next element from same list if exists
            if elem_idx + 1 < len(lists[list_idx]):
                next_value = lists[list_idx][elem_idx + 1]
                heapq.heappush(min_heap, (next_value, list_idx, elem_idx + 1))
                print(f"   Added next element: {next_value} from list {list_idx}")
            
            print(f"   Heap after step: {min_heap}")
            print()
            step += 1
        
        print(f"Final merged list: {result}")
        return result
    
    def merge_k_sorted_arrays_optimized(self, arrays: List[List[int]]) -> List[int]:
        """
        Optimized merge using divide and conquer approach
        
        Time: O(n log k), Space: O(log k) for recursion
        Alternative to heap-based approach
        """
        print(f"Merging {len(arrays)} arrays using divide and conquer:")
        
        def merge_two_arrays(arr1: List[int], arr2: List[int]) -> List[int]:
            """Merge two sorted arrays"""
            result = []
            i = j = 0
            
            while i < len(arr1) and j < len(arr2):
                if arr1[i] <= arr2[j]:
                    result.append(arr1[i])
                    i += 1
                else:
                    result.append(arr2[j])
                    j += 1
            
            # Add remaining elements
            result.extend(arr1[i:])
            result.extend(arr2[j:])
            
            return result
        
        def merge_helper(arrays: List[List[int]]) -> List[int]:
            """Recursively merge arrays using divide and conquer"""
            if not arrays:
                return []
            if len(arrays) == 1:
                return arrays[0]
            
            mid = len(arrays) // 2
            left = merge_helper(arrays[:mid])
            right = merge_helper(arrays[mid:])
            
            return merge_two_arrays(left, right)
        
        if not arrays:
            return []
        
        # Filter out empty arrays
        non_empty_arrays = [arr for arr in arrays if arr]
        
        result = merge_helper(non_empty_arrays)
        print(f"Final merged result: {result}")
        return result
    
    # ==========================================
    # 3. SLIDING WINDOW MAXIMUM/MINIMUM
    # ==========================================
    
    def sliding_window_maximum_heap(self, nums: List[int], k: int) -> List[int]:
        """
        Sliding window maximum using heap
        
        Company: Amazon, Google
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        Note: Deque solution is more efficient O(n), but heap shows different approach
        """
        if not nums or k <= 0:
            return []
        
        # Max heap storing (value, index) pairs
        max_heap = []
        result = []
        
        print(f"Sliding window maximum: nums={nums}, k={k}")
        print("Using max heap approach")
        print()
        
        for i in range(len(nums)):
            # Add current element to heap (negate for max heap)
            heapq.heappush(max_heap, (-nums[i], i))
            
            print(f"Step {i+1}: Added nums[{i}] = {nums[i]} to heap")
            
            # Remove elements outside current window
            while max_heap and max_heap[0][1] <= i - k:
                removed = heapq.heappop(max_heap)
                print(f"   Removed element outside window: {-removed[0]} at index {removed[1]}")
            
            # If window is complete, record maximum
            if i >= k - 1:
                window_max = -max_heap[0][0]
                result.append(window_max)
                window_start = i - k + 1
                window_elements = nums[window_start:i+1]
                print(f"   Window [{window_start}:{i+1}]: {window_elements} -> Maximum = {window_max}")
            
            print(f"   Current heap top 3: {[(-val, idx) for val, idx in sorted(max_heap)[:3]]}")
            print()
        
        print(f"Sliding window maximums: {result}")
        return result
    
    def sliding_window_median(self, nums: List[int], k: int) -> List[float]:
        """
        Sliding window median using two heaps
        
        Company: Google, Facebook
        Difficulty: Hard
        Time: O(n log k), Space: O(k)
        
        Uses max heap for smaller half, min heap for larger half
        """
        if not nums or k <= 0:
            return []
        
        from collections import deque
        
        def find_median(window: List[int]) -> float:
            """Find median of current window using two heaps"""
            window_sorted = sorted(window)
            n = len(window_sorted)
            if n % 2 == 1:
                return float(window_sorted[n // 2])
            else:
                return (window_sorted[n // 2 - 1] + window_sorted[n // 2]) / 2.0
        
        result = []
        window = deque()
        
        print(f"Sliding window median: nums={nums}, k={k}")
        print()
        
        for i in range(len(nums)):
            # Add current element to window
            window.append(nums[i])
            
            # Remove elements outside window
            if len(window) > k:
                removed = window.popleft()
                print(f"Step {i+1}: Removed {removed}, added {nums[i]}")
            else:
                print(f"Step {i+1}: Added {nums[i]}")
            
            # Calculate median if window is complete
            if len(window) == k:
                median = find_median(list(window))
                result.append(median)
                
                window_list = list(window)
                print(f"   Window: {window_list}")
                print(f"   Median: {median}")
            
            print()
        
        print(f"Sliding window medians: {result}")
        return result
    
    # ==========================================
    # 4. GRAPH ALGORITHMS USING HEAPS
    # ==========================================
    
    def dijkstra_shortest_path(self, graph: Dict[str, List[Tuple[str, int]]], 
                              start: str, end: str) -> Tuple[int, List[str]]:
        """
        Dijkstra's shortest path algorithm using min heap
        
        Company: Google, Amazon, Uber, Maps applications
        Difficulty: Hard
        Time: O((V + E) log V), Space: O(V)
        
        Args:
            graph: Adjacency list {node: [(neighbor, weight), ...]}
            start: Starting node
            end: Target node
            
        Returns:
            (shortest_distance, path)
        """
        # Min heap: (distance, node, path)
        min_heap = [(0, start, [start])]
        visited = set()
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        
        print(f"Dijkstra's algorithm: Finding shortest path from '{start}' to '{end}'")
        print(f"Graph: {dict(graph)}")
        print()
        
        step = 1
        while min_heap:
            current_distance, current_node, path = heapq.heappop(min_heap)
            
            print(f"Step {step}: Processing node '{current_node}' (distance: {current_distance})")
            print(f"   Current path: {' -> '.join(path)}")
            
            # Skip if already visited
            if current_node in visited:
                print(f"   Node '{current_node}' already visited, skipping")
                continue
            
            # Mark as visited
            visited.add(current_node)
            
            # Check if reached destination
            if current_node == end:
                print(f"   ✓ Reached destination '{end}'!")
                print(f"   Shortest distance: {current_distance}")
                print(f"   Shortest path: {' -> '.join(path)}")
                return current_distance, path
            
            # Explore neighbors
            neighbors = graph.get(current_node, [])
            print(f"   Neighbors: {neighbors}")
            
            for neighbor, weight in neighbors:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        new_path = path + [neighbor]
                        heapq.heappush(min_heap, (new_distance, neighbor, new_path))
                        
                        print(f"     Updated distance to '{neighbor}': {new_distance}")
                        print(f"     New path: {' -> '.join(new_path)}")
                    else:
                        print(f"     Distance to '{neighbor}' ({new_distance}) not better than current ({distances[neighbor]})")
            
            print(f"   Heap size after processing: {len(min_heap)}")
            print()
            step += 1
        
        print(f"No path found from '{start}' to '{end}'")
        return float('inf'), []
    
    def minimum_spanning_tree_prim(self, graph: Dict[str, List[Tuple[str, int]]]) -> List[Tuple[str, str, int]]:
        """
        Prim's algorithm for Minimum Spanning Tree using min heap
        
        Company: Network design, Google, Amazon
        Difficulty: Hard
        Time: O(E log V), Space: O(V)
        
        Returns list of edges in MST: [(node1, node2, weight), ...]
        """
        if not graph:
            return []
        
        # Start with arbitrary node
        start_node = next(iter(graph))
        mst = []
        visited = {start_node}
        total_weight = 0
        
        # Min heap: (weight, node1, node2)
        min_heap = []
        
        print(f"Prim's MST algorithm starting from node '{start_node}'")
        print(f"Graph: {dict(graph)}")
        print()
        
        # Add all edges from start node to heap
        for neighbor, weight in graph[start_node]:
            heapq.heappush(min_heap, (weight, start_node, neighbor))
        
        print(f"Initial edges from '{start_node}': {[(w, n2) for w, n1, n2 in min_heap]}")
        print()
        
        step = 1
        while min_heap and len(visited) < len(graph):
            weight, node1, node2 = heapq.heappop(min_heap)
            
            print(f"Step {step}: Considering edge ({node1}, {node2}) with weight {weight}")
            
            # Skip if both nodes already in MST (would create cycle)
            if node2 in visited:
                print(f"   Both nodes already in MST, skipping (would create cycle)")
                continue
            
            # Add edge to MST
            mst.append((node1, node2, weight))
            visited.add(node2)
            total_weight += weight
            
            print(f"   ✓ Added edge ({node1}, {node2}) with weight {weight}")
            print(f"   Visited nodes: {visited}")
            print(f"   Current MST weight: {total_weight}")
            
            # Add all edges from new node to heap
            for neighbor, edge_weight in graph[node2]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (edge_weight, node2, neighbor))
                    print(f"   Added edge ({node2}, {neighbor}) with weight {edge_weight} to heap")
            
            print(f"   Heap size: {len(min_heap)}")
            print()
            step += 1
        
        print(f"Minimum Spanning Tree:")
        for node1, node2, weight in mst:
            print(f"   {node1} -- {node2} (weight: {weight})")
        print(f"Total MST weight: {total_weight}")
        
        return mst
    
    # ==========================================
    # 5. SPECIALIZED HEAP APPLICATIONS
    # ==========================================
    
    def task_scheduler_heap(self, tasks: List[str], n: int) -> int:
        """
        Task Scheduler using heap (cooling time between same tasks)
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(m log 26), Space: O(1) where m is total execution time
        
        Args:
            tasks: List of task types (characters)
            n: Cooling time between same tasks
            
        Returns: Minimum time needed to execute all tasks
        """
        # Count task frequencies
        task_count = Counter(tasks)
        
        print(f"Task Scheduler: tasks={tasks}, cooling_time={n}")
        print(f"Task frequencies: {dict(task_count)}")
        print()
        
        # Max heap of task frequencies (negate for max heap)
        max_heap = [-count for count in task_count.values()]
        heapq.heapify(max_heap)
        
        time = 0
        execution_log = []
        
        while max_heap:
            temp_storage = []
            cycle_time = 0
            
            print(f"Time {time}: Starting new cycle")
            print(f"   Available tasks (frequencies): {[-x for x in max_heap]}")
            
            # Execute tasks for n+1 slots (or until no tasks left)
            for i in range(n + 1):
                if max_heap:
                    # Execute most frequent task
                    freq = -heapq.heappop(max_heap)
                    task_name = f"Task{freq}"  # Simplified task naming
                    execution_log.append(task_name)
                    
                    print(f"   Slot {i}: Execute {task_name} (frequency was {freq})")
                    
                    # Decrease frequency and store for later
                    if freq > 1:
                        temp_storage.append(-(freq - 1))
                        print(f"     Task frequency reduced to {freq - 1}")
                    
                    cycle_time += 1
                else:
                    # No tasks available, must idle
                    if temp_storage:  # Only idle if there are more tasks to do
                        execution_log.append("idle")
                        print(f"   Slot {i}: Idle (cooling period)")
                        cycle_time += 1
            
            # Add tasks back to heap
            for task_freq in temp_storage:
                heapq.heappush(max_heap, task_freq)
            
            time += cycle_time
            print(f"   Cycle completed, total time: {time}")
            print(f"   Remaining tasks: {[-x for x in max_heap]}")
            print()
        
        print(f"Task execution order: {execution_log}")
        print(f"Total execution time: {time}")
        
        return time
    
    def find_median_data_stream(self) -> None:
        """
        Find median in data stream using two heaps
        
        Company: Google, Facebook, Amazon
        Difficulty: Hard
        Time: O(log n) per insertion, O(1) for median
        Space: O(n)
        
        Uses max heap for smaller half, min heap for larger half
        """
        
        class MedianFinder:
            def __init__(self):
                self.max_heap = []  # Smaller half (max heap using negation)
                self.min_heap = []  # Larger half (min heap)
            
            def add_number(self, num: int) -> None:
                """Add number to data structure"""
                print(f"Adding number: {num}")
                
                # Add to appropriate heap
                if not self.max_heap or num <= -self.max_heap[0]:
                    heapq.heappush(self.max_heap, -num)
                    print(f"   Added to max_heap (smaller half)")
                else:
                    heapq.heappush(self.min_heap, num)
                    print(f"   Added to min_heap (larger half)")
                
                # Balance heaps
                self._balance_heaps()
                
                print(f"   Max heap (smaller): {[-x for x in self.max_heap]}")
                print(f"   Min heap (larger): {self.min_heap}")
                print(f"   Current median: {self.find_median()}")
                print()
            
            def find_median(self) -> float:
                """Find median of current numbers"""
                if len(self.max_heap) == len(self.min_heap):
                    if not self.max_heap:
                        return 0.0
                    return (-self.max_heap[0] + self.min_heap[0]) / 2.0
                elif len(self.max_heap) > len(self.min_heap):
                    return float(-self.max_heap[0])
                else:
                    return float(self.min_heap[0])
            
            def _balance_heaps(self) -> None:
                """Balance heap sizes (difference ≤ 1)"""
                if len(self.max_heap) > len(self.min_heap) + 1:
                    # Move from max_heap to min_heap
                    value = -heapq.heappop(self.max_heap)
                    heapq.heappush(self.min_heap, value)
                    print(f"   Balanced: moved {value} from max_heap to min_heap")
                elif len(self.min_heap) > len(self.max_heap) + 1:
                    # Move from min_heap to max_heap
                    value = heapq.heappop(self.min_heap)
                    heapq.heappush(self.max_heap, -value)
                    print(f"   Balanced: moved {value} from min_heap to max_heap")
        
        print("=== MEDIAN IN DATA STREAM ===")
        median_finder = MedianFinder()
        
        # Test with stream of numbers
        numbers = [5, 15, 1, 3, 8, 12, 2]
        print(f"Processing stream: {numbers}")
        print()
        
        for num in numbers:
            median_finder.add_number(num)
        
        return median_finder


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_heap_applications():
    """Demonstrate all heap applications and algorithms"""
    print("=== HEAP APPLICATIONS DEMONSTRATION ===\n")
    
    apps = HeapApplications()
    
    # 1. K-element problems
    print("=== K-ELEMENT PROBLEMS ===")
    
    print("1. K Largest Elements:")
    test_array = [3, 2, 1, 5, 6, 4, 7, 9, 8]
    apps.k_largest_elements(test_array, 3)
    print("\n" + "-"*50 + "\n")
    
    print("2. Kth Largest Element:")
    apps.kth_largest_element([3, 2, 3, 1, 2, 4, 5, 5, 6], 4)
    print("\n" + "-"*50 + "\n")
    
    print("3. Top K Frequent Elements:")
    apps.top_k_frequent_elements([1, 1, 1, 2, 2, 3], 2)
    print("\n" + "="*60 + "\n")
    
    # 2. Merge algorithms
    print("=== MERGE ALGORITHMS ===")
    
    print("1. Merge K Sorted Lists:")
    sorted_lists = [
        [1, 4, 5],
        [1, 3, 4],
        [2, 6]
    ]
    apps.merge_k_sorted_lists(sorted_lists)
    print("\n" + "-"*50 + "\n")
    
    print("2. Merge K Sorted Arrays (Divide & Conquer):")
    apps.merge_k_sorted_arrays_optimized(sorted_lists)
    print("\n" + "="*60 + "\n")
    
    # 3. Sliding window problems
    print("=== SLIDING WINDOW PROBLEMS ===")
    
    print("1. Sliding Window Maximum (Heap approach):")
    apps.sliding_window_maximum_heap([1, 3, -1, -3, 5, 3, 6, 7], 3)
    print("\n" + "-"*50 + "\n")
    
    print("2. Sliding Window Median:")
    apps.sliding_window_median([1, 3, -1, -3, 5, 3, 6, 7], 4)
    print("\n" + "="*60 + "\n")
    
    # 4. Graph algorithms
    print("=== GRAPH ALGORITHMS ===")
    
    print("1. Dijkstra's Shortest Path:")
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': []
    }
    apps.dijkstra_shortest_path(graph, 'A', 'E')
    print("\n" + "-"*50 + "\n")
    
    print("2. Minimum Spanning Tree (Prim's Algorithm):")
    mst_graph = {
        'A': [('B', 2), ('C', 3)],
        'B': [('A', 2), ('C', 1), ('D', 1)],
        'C': [('A', 3), ('B', 1), ('D', 4)],
        'D': [('B', 1), ('C', 4)]
    }
    apps.minimum_spanning_tree_prim(mst_graph)
    print("\n" + "="*60 + "\n")
    
    # 5. Specialized applications
    print("=== SPECIALIZED APPLICATIONS ===")
    
    print("1. Task Scheduler:")
    apps.task_scheduler_heap(['A','A','A','B','B','B'], 2)
    print("\n" + "-"*50 + "\n")
    
    print("2. Median in Data Stream:")
    apps.find_median_data_stream()


if __name__ == "__main__":
    demonstrate_heap_applications()
    
    print("\n=== HEAP APPLICATIONS MASTERY GUIDE ===")
    
    print("\n🎯 HEAP APPLICATION CATEGORIES:")
    print("• K-Element Problems: Finding top/bottom k elements efficiently")
    print("• Merge Operations: Combining multiple sorted sequences")
    print("• Sliding Window: Maintaining extremes in moving windows")
    print("• Graph Algorithms: Shortest paths, minimum spanning trees")
    print("• Streaming Data: Online algorithms for continuous data")
    
    print("\n📊 COMPLEXITY PATTERNS:")
    print("• K-element problems: O(n log k) time, O(k) space")
    print("• Merge k sequences: O(n log k) where n is total elements")
    print("• Graph algorithms: O((V+E) log V) for Dijkstra/Prim")
    print("• Streaming median: O(log n) per insertion")
    print("• Sliding window: O(n log k) with heap vs O(n) with deque")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use min heap for k largest, max heap for k smallest")
    print("• Two heaps for median: max heap (smaller half) + min heap (larger half)")
    print("• Lazy deletion for sliding window problems")
    print("• Custom comparators for complex objects")
    print("• Consider alternatives: deque for sliding window maximum")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Python heapq is min heap - negate values for max heap")
    print("• Store (priority, data) tuples for custom ordering")
    print("• Handle edge cases: empty inputs, k larger than array size")
    print("• Balance heap sizes for median finding")
    print("• Use stable sorting for tie-breaking")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Search engines: Top k relevant results")
    print("• Social media: Top trending posts")
    print("• Stock trading: Price monitoring and alerts")
    print("• Network routing: Shortest path algorithms")
    print("• Task scheduling: Priority-based job queues")
    print("• Data analysis: Streaming statistics and percentiles")
    
    print("\n🎓 PROBLEM-SOLVING PATTERNS:")
    print("• 'Top k' or 'bottom k' → Use heap of size k")
    print("• 'Merge sorted' → Use min heap with pointers")
    print("• 'Sliding window extremes' → Consider heap vs deque")
    print("• 'Shortest path' → Dijkstra with min heap")
    print("• 'Continuous median' → Two heaps (max + min)")
    print("• 'Task scheduling' → Max heap with frequency/priority")
