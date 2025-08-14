"""
Queue Applications and Algorithms
=================================

Topics: Advanced queue algorithms, BFS, level-order traversal, scheduling
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber
Difficulty: Medium to Hard
Time Complexity: Often O(n) with queue optimization
Space Complexity: O(n) for queue storage
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque, defaultdict
import heapq
import time

class QueueAlgorithms:
    
    def __init__(self):
        """Initialize with solution tracking"""
        self.solution_steps = []
        self.visited_nodes = set()
    
    # ==========================================
    # 1. GRAPH ALGORITHMS USING QUEUE
    # ==========================================
    
    def breadth_first_search(self, graph: Dict[int, List[int]], start: int) -> List[int]:
        """
        Comprehensive BFS implementation with detailed tracking
        
        Company: Google, Amazon, Microsoft, Facebook
        Difficulty: Medium
        Time: O(V + E), Space: O(V)
        
        Returns nodes in BFS order
        """
        visited = set()
        queue = deque([start])
        bfs_order = []
        
        print(f"BFS traversal starting from node {start}")
        print(f"Graph adjacency list: {dict(graph)}")
        print()
        
        step = 1
        while queue:
            print(f"Step {step}:")
            print(f"   Queue: {list(queue)}")
            print(f"   Visited so far: {visited}")
            
            # Dequeue next node
            current = queue.popleft()
            
            if current not in visited:
                # Mark as visited and add to result
                visited.add(current)
                bfs_order.append(current)
                print(f"   Visiting node {current}")
                print(f"   BFS order: {bfs_order}")
                
                # Add unvisited neighbors to queue
                neighbors = graph.get(current, [])
                print(f"   Neighbors of {current}: {neighbors}")
                
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
                        print(f"     Added {neighbor} to queue")
                
                print(f"   Queue after adding neighbors: {list(queue)}")
            else:
                print(f"   Node {current} already visited, skipping")
            
            print()
            step += 1
        
        print(f"BFS traversal complete: {bfs_order}")
        print(f"Total nodes visited: {len(visited)}")
        return bfs_order
    
    def shortest_path_unweighted(self, graph: Dict[int, List[int]], 
                                start: int, end: int) -> Tuple[List[int], int]:
        """
        Find shortest path in unweighted graph using BFS
        
        Company: Amazon, Google
        Difficulty: Medium
        Time: O(V + E), Space: O(V)
        
        Returns (path, distance)
        """
        if start == end:
            return [start], 0
        
        visited = set()
        queue = deque([(start, [start])])  # (node, path_to_node)
        
        print(f"Finding shortest path from {start} to {end}")
        print(f"Graph: {dict(graph)}")
        print()
        
        step = 1
        while queue:
            print(f"Step {step}: Queue size = {len(queue)}")
            
            current, path = queue.popleft()
            print(f"   Processing node {current}, path: {path}")
            
            if current in visited:
                print(f"   Node {current} already visited, skipping")
                continue
            
            visited.add(current)
            
            # Check if we reached the destination
            if current == end:
                print(f"   ✓ Reached destination {end}!")
                print(f"   Shortest path: {path}")
                print(f"   Distance: {len(path) - 1}")
                return path, len(path) - 1
            
            # Explore neighbors
            neighbors = graph.get(current, [])
            print(f"   Neighbors: {neighbors}")
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
                    print(f"     Added {neighbor} with path: {new_path}")
            
            print(f"   Queue after processing: {len(queue)} items")
            step += 1
            print()
        
        print(f"No path found from {start} to {end}")
        return [], -1
    
    def connected_components(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        """
        Find all connected components using BFS
        
        Company: Facebook, Google
        Difficulty: Medium
        Time: O(V + E), Space: O(V)
        """
        visited = set()
        components = []
        all_nodes = set(graph.keys())
        
        # Add isolated nodes
        for node_list in graph.values():
            all_nodes.update(node_list)
        
        print(f"Finding connected components in graph: {dict(graph)}")
        print(f"All nodes: {sorted(all_nodes)}")
        print()
        
        component_num = 1
        for node in sorted(all_nodes):
            if node not in visited:
                print(f"Component {component_num}: Starting BFS from node {node}")
                
                # BFS for current component
                queue = deque([node])
                current_component = []
                
                while queue:
                    current = queue.popleft()
                    
                    if current not in visited:
                        visited.add(current)
                        current_component.append(current)
                        print(f"   Added {current} to component")
                        
                        # Add unvisited neighbors
                        neighbors = graph.get(current, [])
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(sorted(current_component))
                print(f"   Component {component_num}: {sorted(current_component)}")
                print()
                component_num += 1
        
        print(f"Total connected components: {len(components)}")
        for i, component in enumerate(components, 1):
            print(f"   Component {i}: {component}")
        
        return components
    
    # ==========================================
    # 2. TREE ALGORITHMS USING QUEUE
    # ==========================================
    
    def level_order_traversal(self, tree: Dict[str, List[str]], root: str) -> List[List[str]]:
        """
        Level-order (breadth-first) tree traversal
        
        Company: Amazon, Microsoft, Apple
        Difficulty: Medium
        Time: O(n), Space: O(w) where w is maximum width
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        print(f"Level-order traversal starting from root '{root}'")
        print(f"Tree structure: {dict(tree)}")
        print()
        
        level = 1
        while queue:
            level_size = len(queue)
            current_level = []
            
            print(f"Level {level}: Processing {level_size} nodes")
            print(f"   Queue at start: {list(queue)}")
            
            # Process all nodes at current level
            for i in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                print(f"   Processing node '{node}'")
                
                # Add children to queue for next level
                children = tree.get(node, [])
                for child in children:
                    queue.append(child)
                    print(f"     Added child '{child}' to queue")
            
            result.append(current_level)
            print(f"   Level {level} completed: {current_level}")
            print(f"   Queue for next level: {list(queue)}")
            print()
            level += 1
        
        print(f"Level-order traversal result: {result}")
        return result
    
    def right_side_view(self, tree: Dict[str, List[str]], root: str) -> List[str]:
        """
        Get right side view of binary tree using level-order traversal
        
        Company: Facebook, Amazon
        Difficulty: Medium
        Time: O(n), Space: O(w)
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        print(f"Finding right side view of tree from root '{root}'")
        print(f"Tree structure: {dict(tree)}")
        print()
        
        level = 1
        while queue:
            level_size = len(queue)
            
            print(f"Level {level}: {level_size} nodes in queue: {list(queue)}")
            
            for i in range(level_size):
                node = queue.popleft()
                
                # If this is the last node in current level, add to result
                if i == level_size - 1:
                    result.append(node)
                    print(f"   Rightmost node at level {level}: '{node}' ✓")
                else:
                    print(f"   Processing node '{node}'")
                
                # Add children to queue
                children = tree.get(node, [])
                for child in children:
                    queue.append(child)
            
            level += 1
            print()
        
        print(f"Right side view: {result}")
        return result
    
    def zigzag_level_order(self, tree: Dict[str, List[str]], root: str) -> List[List[str]]:
        """
        Zigzag (spiral) level order traversal
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        Time: O(n), Space: O(w)
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        left_to_right = True
        
        print(f"Zigzag level-order traversal from root '{root}'")
        print(f"Tree structure: {dict(tree)}")
        print()
        
        level = 1
        while queue:
            level_size = len(queue)
            current_level = []
            
            direction = "left-to-right" if left_to_right else "right-to-left"
            print(f"Level {level}: {level_size} nodes, direction: {direction}")
            print(f"   Queue: {list(queue)}")
            
            # Process all nodes at current level
            for i in range(level_size):
                node = queue.popleft()
                current_level.append(node)
                
                # Add children to queue
                children = tree.get(node, [])
                for child in children:
                    queue.append(child)
            
            # Reverse if going right to left
            if not left_to_right:
                current_level.reverse()
                print(f"   Reversed order: {current_level}")
            else:
                print(f"   Normal order: {current_level}")
            
            result.append(current_level)
            left_to_right = not left_to_right  # Toggle direction
            
            print(f"   Next level direction: {'left-to-right' if left_to_right else 'right-to-left'}")
            print()
            level += 1
        
        print(f"Zigzag traversal result: {result}")
        return result
    
    # ==========================================
    # 3. SCHEDULING ALGORITHMS
    # ==========================================
    
    def round_robin_scheduling(self, processes: List[Tuple[str, int]], time_quantum: int) -> List[Tuple[str, int, int]]:
        """
        Round Robin CPU Scheduling using Queue
        
        Company: Operating Systems companies, System design interviews
        Difficulty: Medium
        Time: O(n * total_time / quantum), Space: O(n)
        
        Args:
            processes: List of (process_name, burst_time)
            time_quantum: Time slice for each process
            
        Returns:
            List of (process_name, start_time, end_time)
        """
        queue = deque()
        remaining_time = {}
        completed = []
        current_time = 0
        
        # Initialize queue and remaining times
        for process_name, burst_time in processes:
            queue.append(process_name)
            remaining_time[process_name] = burst_time
        
        print(f"Round Robin Scheduling (Time Quantum: {time_quantum})")
        print(f"Processes: {processes}")
        print()
        
        round_num = 1
        while queue:
            print(f"Round {round_num}:")
            print(f"   Queue: {list(queue)}")
            print(f"   Remaining times: {remaining_time}")
            
            # Get next process
            current_process = queue.popleft()
            remaining = remaining_time[current_process]
            
            print(f"   Current process: {current_process}")
            print(f"   Remaining time: {remaining}")
            
            # Execute for time quantum or remaining time, whichever is smaller
            execution_time = min(time_quantum, remaining)
            start_time = current_time
            current_time += execution_time
            
            print(f"   Executing for {execution_time} units (from {start_time} to {current_time})")
            
            # Update remaining time
            remaining_time[current_process] -= execution_time
            
            if remaining_time[current_process] == 0:
                # Process completed
                completed.append((current_process, start_time, current_time))
                print(f"   Process {current_process} completed!")
            else:
                # Add back to queue for next round
                queue.append(current_process)
                print(f"   Process {current_process} added back to queue (remaining: {remaining_time[current_process]})")
            
            print(f"   Current time: {current_time}")
            print()
            round_num += 1
        
        print("Scheduling completed!")
        print("Process execution timeline:")
        for process, start, end in completed:
            print(f"   {process}: {start} -> {end} (duration: {end - start})")
        
        return completed
    
    def priority_scheduling(self, tasks: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """
        Priority-based task scheduling using priority queue
        
        Company: System design, OS concepts
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Args:
            tasks: List of (task_name, priority, burst_time)
        
        Returns:
            List of (task_name, start_time, end_time)
        """
        # Priority queue (min-heap, so negate priority for max-heap behavior)
        pq = []
        completed = []
        current_time = 0
        
        print("Priority-based Task Scheduling")
        print(f"Tasks (name, priority, burst_time): {tasks}")
        print("Higher priority number = higher priority")
        print()
        
        # Add all tasks to priority queue
        for task_name, priority, burst_time in tasks:
            # Use negative priority for max-heap behavior
            heapq.heappush(pq, (-priority, task_name, burst_time))
        
        print("Priority queue (sorted by priority):")
        temp_pq = pq[:]
        task_num = 1
        while temp_pq:
            neg_priority, name, burst = heapq.heappop(temp_pq)
            print(f"   {task_num}. {name} (priority: {-neg_priority}, burst: {burst})")
            task_num += 1
        print()
        
        # Execute tasks in priority order
        execution_order = 1
        while pq:
            neg_priority, task_name, burst_time = heapq.heappop(pq)
            priority = -neg_priority
            
            start_time = current_time
            current_time += burst_time
            
            completed.append((task_name, start_time, current_time))
            
            print(f"Execution {execution_order}:")
            print(f"   Task: {task_name}")
            print(f"   Priority: {priority}")
            print(f"   Execution: {start_time} -> {current_time} (duration: {burst_time})")
            print(f"   Remaining tasks: {len(pq)}")
            print()
            execution_order += 1
        
        print("Priority scheduling completed!")
        print("Execution timeline:")
        for task, start, end in completed:
            print(f"   {task}: {start} -> {end}")
        
        return completed
    
    # ==========================================
    # 4. SLIDING WINDOW WITH QUEUE
    # ==========================================
    
    def sliding_window_maximum_deque(self, nums: List[int], k: int) -> List[int]:
        """
        Sliding window maximum using deque (optimized queue)
        
        Company: Amazon, Google, Microsoft
        Difficulty: Hard
        Time: O(n), Space: O(k)
        
        Uses monotonic deque to maintain maximum in each window
        """
        if not nums or k == 0:
            return []
        
        dq = deque()  # Store indices
        result = []
        
        print(f"Finding sliding window maximum: nums={nums}, k={k}")
        print("Using monotonic deque for O(n) solution")
        print()
        
        for i in range(len(nums)):
            print(f"Step {i+1}: Processing nums[{i}] = {nums[i]}")
            
            # Remove indices outside current window
            while dq and dq[0] <= i - k:
                removed_idx = dq.popleft()
                print(f"   Remove index {removed_idx} (outside window)")
            
            # Remove indices with smaller values (they can't be maximum)
            while dq and nums[dq[-1]] < nums[i]:
                removed_idx = dq.pop()
                print(f"   Remove index {removed_idx} (nums[{removed_idx}]={nums[removed_idx]} < {nums[i]})")
            
            # Add current index
            dq.append(i)
            print(f"   Add index {i}")
            print(f"   Deque indices: {list(dq)}")
            print(f"   Deque values: {[nums[idx] for idx in dq]}")
            
            # If window is complete, record maximum
            if i >= k - 1:
                maximum = nums[dq[0]]
                result.append(maximum)
                window_start = i - k + 1
                print(f"   Window [{window_start}:{i+1}]: {nums[window_start:i+1]} -> Maximum = {maximum}")
            
            print()
        
        print(f"Sliding window maximums: {result}")
        return result
    
    def first_negative_in_window(self, nums: List[int], k: int) -> List[int]:
        """
        Find first negative number in each sliding window
        
        Company: Amazon, Microsoft
        Difficulty: Medium
        Time: O(n), Space: O(k)
        """
        negative_queue = deque()  # Store indices of negative numbers
        result = []
        
        print(f"Finding first negative in sliding windows: nums={nums}, k={k}")
        print()
        
        for i in range(len(nums)):
            print(f"Step {i+1}: Processing nums[{i}] = {nums[i]}")
            
            # Remove indices outside current window
            while negative_queue and negative_queue[0] <= i - k:
                removed_idx = negative_queue.popleft()
                print(f"   Remove index {removed_idx} (outside window)")
            
            # Add current index if number is negative
            if nums[i] < 0:
                negative_queue.append(i)
                print(f"   Added negative number at index {i}")
            
            print(f"   Negative queue indices: {list(negative_queue)}")
            print(f"   Negative queue values: {[nums[idx] for idx in negative_queue]}")
            
            # If window is complete, find first negative
            if i >= k - 1:
                window_start = i - k + 1
                if negative_queue:
                    first_negative = nums[negative_queue[0]]
                    result.append(first_negative)
                    print(f"   Window [{window_start}:{i+1}]: First negative = {first_negative}")
                else:
                    result.append(0)  # No negative number
                    print(f"   Window [{window_start}:{i+1}]: No negative number (0)")
            
            print()
        
        print(f"First negatives in windows: {result}")
        return result
    
    # ==========================================
    # 5. ADVANCED QUEUE PROBLEMS
    # ==========================================
    
    def rotting_oranges(self, grid: List[List[int]]) -> int:
        """
        Rotting Oranges problem using multi-source BFS
        
        Company: Amazon, Facebook, Google
        Difficulty: Medium
        Time: O(m*n), Space: O(m*n)
        
        0 = empty, 1 = fresh orange, 2 = rotten orange
        Find minimum time for all oranges to rot
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        # Find all initial rotten oranges and count fresh ones
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))  # (row, col, time)
                elif grid[i][j] == 1:
                    fresh_count += 1
        
        print("Rotting Oranges Problem:")
        print("Grid (0=empty, 1=fresh, 2=rotten):")
        for row in grid:
            print(f"   {row}")
        print(f"Initial fresh oranges: {fresh_count}")
        print(f"Initial rotten oranges: {len(queue)}")
        print()
        
        if fresh_count == 0:
            print("No fresh oranges to rot!")
            return 0
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        max_time = 0
        
        step = 1
        while queue:
            row, col, time = queue.popleft()
            max_time = max(max_time, time)
            
            print(f"Step {step}: Processing rotten orange at ({row}, {col}) at time {time}")
            
            # Spread to adjacent fresh oranges
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < rows and 0 <= new_col < cols and 
                    grid[new_row][new_col] == 1):
                    
                    # Fresh orange becomes rotten
                    grid[new_row][new_col] = 2
                    fresh_count -= 1
                    queue.append((new_row, new_col, time + 1))
                    
                    print(f"   Orange at ({new_row}, {new_col}) becomes rotten at time {time + 1}")
                    print(f"   Remaining fresh oranges: {fresh_count}")
            
            step += 1
        
        print(f"\nFinal grid:")
        for row in grid:
            print(f"   {row}")
        
        if fresh_count > 0:
            print(f"Cannot rot all oranges! {fresh_count} oranges remain fresh.")
            return -1
        
        print(f"All oranges rotted in {max_time} minutes!")
        return max_time
    
    def walls_and_gates(self, rooms: List[List[int]]) -> None:
        """
        Fill each empty room with distance to nearest gate
        
        Company: Google, Facebook
        Difficulty: Medium
        Time: O(m*n), Space: O(m*n)
        
        -1 = wall, 0 = gate, INF = empty room
        """
        if not rooms or not rooms[0]:
            return
        
        INF = 2147483647
        rows, cols = len(rooms), len(rooms[0])
        queue = deque()
        
        # Find all gates
        for i in range(rows):
            for j in range(cols):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        print("Walls and Gates Problem:")
        print("Grid (-1=wall, 0=gate, 2147483647=empty room):")
        for row in rooms:
            print(f"   {row}")
        print(f"Found {len(queue)} gates")
        print()
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        step = 1
        while queue:
            row, col = queue.popleft()
            
            print(f"Step {step}: Processing position ({row}, {col}) with distance {rooms[row][col]}")
            
            # Explore all 4 directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < rows and 0 <= new_col < cols and 
                    rooms[new_row][new_col] == INF):
                    
                    # Update distance and add to queue
                    rooms[new_row][new_col] = rooms[row][col] + 1
                    queue.append((new_row, new_col))
                    
                    print(f"   Updated ({new_row}, {new_col}) to distance {rooms[new_row][new_col]}")
            
            step += 1
        
        print(f"\nFinal grid with distances to nearest gate:")
        for row in rooms:
            print(f"   {row}")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_queue_algorithms():
    """Demonstrate all queue algorithms and applications"""
    print("=== QUEUE ALGORITHMS DEMONSTRATION ===\n")
    
    algorithms = QueueAlgorithms()
    
    # 1. Graph algorithms
    print("=== GRAPH ALGORITHMS ===")
    
    print("1. Breadth-First Search:")
    graph = {
        1: [2, 3],
        2: [1, 4, 5],
        3: [1, 6],
        4: [2],
        5: [2, 6],
        6: [3, 5]
    }
    algorithms.breadth_first_search(graph, 1)
    print("\n" + "-"*60 + "\n")
    
    print("2. Shortest Path in Unweighted Graph:")
    path, distance = algorithms.shortest_path_unweighted(graph, 1, 6)
    print("\n" + "-"*60 + "\n")
    
    print("3. Connected Components:")
    disconnected_graph = {
        1: [2], 2: [1], 3: [4], 4: [3], 5: []
    }
    algorithms.connected_components(disconnected_graph)
    print("\n" + "="*60 + "\n")
    
    # 2. Tree algorithms
    print("=== TREE ALGORITHMS ===")
    
    tree = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': [], 'E': ['H'], 'F': [], 'G': [], 'H': []
    }
    
    print("1. Level-order Traversal:")
    algorithms.level_order_traversal(tree, 'A')
    print("\n" + "-"*60 + "\n")
    
    print("2. Right Side View:")
    algorithms.right_side_view(tree, 'A')
    print("\n" + "-"*60 + "\n")
    
    print("3. Zigzag Level Order:")
    algorithms.zigzag_level_order(tree, 'A')
    print("\n" + "="*60 + "\n")
    
    # 3. Scheduling algorithms
    print("=== SCHEDULING ALGORITHMS ===")
    
    print("1. Round Robin Scheduling:")
    processes = [("P1", 10), ("P2", 5), ("P3", 8)]
    algorithms.round_robin_scheduling(processes, 3)
    print("\n" + "-"*60 + "\n")
    
    print("2. Priority Scheduling:")
    tasks = [("Task1", 3, 4), ("Task2", 1, 2), ("Task3", 4, 1), ("Task4", 2, 3)]
    algorithms.priority_scheduling(tasks)
    print("\n" + "="*60 + "\n")
    
    # 4. Sliding window problems
    print("=== SLIDING WINDOW PROBLEMS ===")
    
    print("1. Sliding Window Maximum:")
    algorithms.sliding_window_maximum_deque([1, 3, -1, -3, 5, 3, 6, 7], 3)
    print("\n" + "-"*60 + "\n")
    
    print("2. First Negative in Window:")
    algorithms.first_negative_in_window([12, -1, -7, 8, -15, 30, 16, 28], 3)
    print("\n" + "="*60 + "\n")
    
    # 5. Advanced problems
    print("=== ADVANCED QUEUE PROBLEMS ===")
    
    print("1. Rotting Oranges:")
    orange_grid = [
        [2, 1, 1],
        [1, 1, 0],
        [0, 1, 1]
    ]
    algorithms.rotting_oranges(orange_grid)
    print("\n" + "-"*60 + "\n")
    
    print("2. Walls and Gates:")
    INF = 2147483647
    rooms = [
        [INF, -1, 0, INF],
        [INF, INF, INF, -1],
        [INF, -1, INF, -1],
        [0, -1, INF, INF]
    ]
    algorithms.walls_and_gates(rooms)


if __name__ == "__main__":
    demonstrate_queue_algorithms()
    
    print("\n=== QUEUE ALGORITHMS MASTERY GUIDE ===")
    
    print("\n🎯 ALGORITHM CATEGORIES:")
    print("• Graph Traversal: BFS, shortest paths, connected components")
    print("• Tree Processing: Level-order, right view, zigzag traversal")
    print("• Scheduling: Round robin, priority-based, task management")
    print("• Sliding Window: Maximum/minimum in windows, pattern matching")
    print("• Multi-source BFS: Shortest distances, spreading problems")
    
    print("\n📋 BFS TEMPLATE:")
    print("1. Initialize queue with starting nodes")
    print("2. Mark visited to avoid cycles")
    print("3. Process current level completely")
    print("4. Add neighbors to queue for next level")
    print("5. Continue until queue is empty")
    
    print("\n⚡ OPTIMIZATION TECHNIQUES:")
    print("• Use deque for efficient front/rear operations")
    print("• Store additional information in queue (distance, time)")
    print("• Multi-source BFS for problems with multiple starting points")
    print("• Monotonic deque for sliding window optimizations")
    print("• Priority queue for scheduling and weighted problems")
    
    print("\n🔍 PROBLEM IDENTIFICATION:")
    print("• 'Level by level' processing → Level-order traversal")
    print("• 'Shortest path' in unweighted graph → BFS")
    print("• 'Minimum time/steps' → Multi-source BFS")
    print("• 'Connected components' → BFS/DFS from each unvisited node")
    print("• 'Sliding window maximum' → Monotonic deque")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Social networks: Friend recommendations, shortest path")
    print("• Operating systems: Process scheduling, resource allocation")
    print("• Web crawling: Breadth-first website exploration")
    print("• Game AI: Shortest path finding, level generation")
    print("• Network routing: Finding optimal paths")
    print("• Cache systems: LRU implementation, buffer management")

