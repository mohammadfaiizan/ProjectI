"""
Priority Queue Applications and Advanced Algorithms
==================================================

Topics: Advanced priority queue algorithms, graph algorithms, optimization problems
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix, Uber
Difficulty: Medium to Hard
Time Complexity: Varies by application, often O(V log V + E) for graphs
Space Complexity: O(V) for graphs, O(n) for general problems
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Callable
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import math
import time

class PriorityQueueApplications:
    
    def __init__(self):
        """Initialize with solution tracking"""
        self.solution_steps = []
        self.algorithm_stats = {}
    
    # ==========================================
    # 1. GRAPH ALGORITHMS WITH PRIORITY QUEUES
    # ==========================================
    
    def dijkstra_shortest_path(self, graph: Dict[str, List[Tuple[str, int]]], 
                              start: str, end: str = None) -> Dict[str, Tuple[int, List[str]]]:
        """
        Dijkstra's Shortest Path Algorithm using Priority Queue
        
        Company: Google, Amazon, Uber, Maps/Navigation apps
        Difficulty: Hard
        Time: O((V + E) log V), Space: O(V)
        
        Args:
            graph: Adjacency list {node: [(neighbor, weight), ...]}
            start: Starting node
            end: Optional target node (if None, finds shortest paths to all nodes)
            
        Returns:
            Dictionary {node: (distance, path)}
        """
        # Priority queue: (distance, node, path)
        pq = [(0, start, [start])]
        distances = {start: 0}
        visited = set()
        shortest_paths = {}
        
        print(f"Dijkstra's Algorithm from '{start}'" + (f" to '{end}'" if end else " to all nodes"))
        print(f"Graph: {dict(graph)}")
        print()
        
        step = 1
        while pq:
            current_dist, current_node, path = heapq.heappop(pq)
            
            print(f"Step {step}: Processing node '{current_node}' (distance: {current_dist})")
            print(f"   Current path: {' -> '.join(path)}")
            
            # Skip if already processed with shorter path
            if current_node in visited:
                print(f"   Already visited '{current_node}' with shorter path, skipping")
                continue
            
            # Mark as visited and record shortest path
            visited.add(current_node)
            shortest_paths[current_node] = (current_dist, path)
            
            print(f"   ✓ Shortest path to '{current_node}': distance {current_dist}")
            
            # Early termination if target found
            if end and current_node == end:
                print(f"   🎯 Reached target '{end}'!")
                break
            
            # Explore neighbors
            neighbors = graph.get(current_node, [])
            print(f"   Neighbors: {neighbors}")
            
            for neighbor, weight in neighbors:
                if neighbor not in visited:
                    new_distance = current_dist + weight
                    
                    # Only add if we found a shorter path
                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        new_path = path + [neighbor]
                        heapq.heappush(pq, (new_distance, neighbor, new_path))
                        
                        print(f"     Updated '{neighbor}': distance {new_distance}, path {' -> '.join(new_path)}")
                    else:
                        print(f"     '{neighbor}': current distance {new_distance} ≥ best known {distances.get(neighbor, float('inf'))}")
            
            print(f"   Queue size: {len(pq)}")
            step += 1
            print()
        
        print("Final shortest paths:")
        for node, (dist, path) in shortest_paths.items():
            print(f"   {start} → {node}: distance {dist}, path {' -> '.join(path)}")
        
        return shortest_paths
    
    def prim_minimum_spanning_tree(self, graph: Dict[str, List[Tuple[str, int]]]) -> List[Tuple[str, str, int]]:
        """
        Prim's Algorithm for Minimum Spanning Tree using Priority Queue
        
        Company: Network design, Infrastructure optimization
        Difficulty: Hard
        Time: O(E log V), Space: O(V)
        
        Returns: List of edges in MST [(node1, node2, weight), ...]
        """
        if not graph:
            return []
        
        # Start with arbitrary node
        start_node = next(iter(graph))
        mst_edges = []
        visited = {start_node}
        total_weight = 0
        
        # Priority queue: (weight, node1, node2)
        pq = []
        
        print(f"Prim's MST Algorithm starting from '{start_node}'")
        print(f"Graph: {dict(graph)}")
        print()
        
        # Add all edges from start node
        for neighbor, weight in graph[start_node]:
            heapq.heappush(pq, (weight, start_node, neighbor))
        
        print(f"Initial edges from '{start_node}': {[(w, n2) for w, n1, n2 in pq]}")
        print()
        
        step = 1
        while pq and len(visited) < len(graph):
            weight, node1, node2 = heapq.heappop(pq)
            
            print(f"Step {step}: Considering edge ({node1}, {node2}) with weight {weight}")
            
            # Skip if both nodes already in MST (would create cycle)
            if node2 in visited:
                print(f"   Both nodes already in MST, skipping (avoids cycle)")
                continue
            
            # Add edge to MST
            mst_edges.append((node1, node2, weight))
            visited.add(node2)
            total_weight += weight
            
            print(f"   ✓ Added edge ({node1}, {node2}) with weight {weight}")
            print(f"   Visited nodes: {sorted(visited)}")
            print(f"   MST total weight: {total_weight}")
            
            # Add all edges from newly added node
            new_edges_added = 0
            for neighbor, edge_weight in graph[node2]:
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, node2, neighbor))
                    new_edges_added += 1
            
            if new_edges_added > 0:
                print(f"   Added {new_edges_added} new edges from '{node2}' to queue")
            
            print(f"   Queue size: {len(pq)}")
            step += 1
            print()
        
        print("Minimum Spanning Tree:")
        for node1, node2, weight in mst_edges:
            print(f"   {node1} -- {node2} (weight: {weight})")
        print(f"Total MST weight: {total_weight}")
        
        return mst_edges
    
    def a_star_pathfinding(self, grid: List[List[int]], start: Tuple[int, int], 
                          goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* Pathfinding Algorithm using Priority Queue
        
        Company: Game development, Robotics, Navigation
        Difficulty: Hard
        Time: O(b^d), Space: O(b^d) where b is branching factor, d is depth
        
        Args:
            grid: 2D grid where 0 = passable, 1 = obstacle
            start: Starting coordinates (row, col)
            goal: Target coordinates (row, col)
            
        Returns: List of coordinates representing the path
        """
        rows, cols = len(grid), len(grid[0])
        
        def heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance heuristic"""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        def get_neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
            """Get valid neighboring positions"""
            row, col = pos
            neighbors = []
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-directional
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < rows and 0 <= new_col < cols and 
                    grid[new_row][new_col] == 0):
                    neighbors.append((new_row, new_col))
            
            return neighbors
        
        # Priority queue: (f_score, g_score, position, path)
        pq = [(heuristic(start), 0, start, [start])]
        visited = set()
        g_scores = {start: 0}
        
        print(f"A* Pathfinding from {start} to {goal}")
        print("Grid (0 = passable, 1 = obstacle):")
        for i, row in enumerate(grid):
            row_str = ""
            for j, cell in enumerate(row):
                if (i, j) == start:
                    row_str += "S "
                elif (i, j) == goal:
                    row_str += "G "
                elif cell == 1:
                    row_str += "# "
                else:
                    row_str += ". "
            print(f"   {row_str}")
        print()
        
        step = 1
        while pq:
            f_score, g_score, current_pos, path = heapq.heappop(pq)
            
            print(f"Step {step}: Exploring {current_pos}")
            print(f"   g_score (actual distance): {g_score}")
            print(f"   h_score (heuristic): {f_score - g_score}")
            print(f"   f_score (total): {f_score}")
            
            if current_pos in visited:
                print(f"   Already visited, skipping")
                continue
            
            visited.add(current_pos)
            
            # Check if reached goal
            if current_pos == goal:
                print(f"   🎯 Reached goal {goal}!")
                print(f"   Final path: {path}")
                print(f"   Path length: {len(path) - 1} steps")
                return path
            
            # Explore neighbors
            neighbors = get_neighbors(current_pos)
            print(f"   Valid neighbors: {neighbors}")
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                
                tentative_g = g_score + 1  # Cost of moving to neighbor
                
                # Only add if this path to neighbor is better
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = heuristic(neighbor)
                    f_score_neighbor = tentative_g + h_score
                    new_path = path + [neighbor]
                    
                    heapq.heappush(pq, (f_score_neighbor, tentative_g, neighbor, new_path))
                    print(f"     Added {neighbor}: g={tentative_g}, h={h_score}, f={f_score_neighbor}")
            
            print()
            step += 1
        
        print("No path found!")
        return []
    
    # ==========================================
    # 2. SCHEDULING AND RESOURCE ALLOCATION
    # ==========================================
    
    def cpu_process_scheduling(self, processes: List[Tuple[str, int, int, int]]) -> List[Tuple[str, int, int]]:
        """
        CPU Process Scheduling using Priority Queue (Shortest Job First with Priorities)
        
        Company: Operating Systems, System Design
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Args:
            processes: List of (process_name, arrival_time, burst_time, priority)
        
        Returns: List of (process_name, start_time, end_time)
        """
        # Priority queue: (priority, arrival_time, burst_time, process_name)
        ready_queue = []
        waiting_processes = sorted(processes, key=lambda x: x[1])  # Sort by arrival time
        schedule = []
        current_time = 0
        process_index = 0
        
        print("CPU Process Scheduling (Priority + Shortest Job First)")
        print("Processes (name, arrival, burst, priority):")
        for name, arrival, burst, priority in processes:
            print(f"   {name}: arrives={arrival}, burst={burst}, priority={priority}")
        print()
        
        while process_index < len(waiting_processes) or ready_queue:
            # Add processes that have arrived to ready queue
            while (process_index < len(waiting_processes) and 
                   waiting_processes[process_index][1] <= current_time):
                
                name, arrival, burst, priority = waiting_processes[process_index]
                # Use priority first, then burst time as tiebreaker
                heapq.heappush(ready_queue, (priority, burst, arrival, name))
                print(f"Time {current_time}: Process {name} added to ready queue")
                process_index += 1
            
            if ready_queue:
                # Execute highest priority process
                priority, burst, arrival, name = heapq.heappop(ready_queue)
                start_time = current_time
                end_time = current_time + burst
                
                schedule.append((name, start_time, end_time))
                current_time = end_time
                
                print(f"Time {start_time}-{end_time}: Executing {name} "
                      f"(priority={priority}, burst={burst})")
                
                # Show remaining processes in queue
                if ready_queue:
                    remaining = [name for _, _, _, name in ready_queue]
                    print(f"   Ready queue: {remaining}")
                print()
            else:
                # No processes ready, advance time to next arrival
                if process_index < len(waiting_processes):
                    current_time = waiting_processes[process_index][1]
                    print(f"CPU idle, advancing to time {current_time}")
        
        print("Final schedule:")
        for name, start, end in schedule:
            print(f"   {name}: {start} -> {end} (duration: {end - start})")
        
        # Calculate metrics
        total_turnaround = 0
        total_waiting = 0
        
        print("\nPerformance metrics:")
        for i, (name, start, end) in enumerate(schedule):
            original_process = processes[i]
            arrival_time = original_process[1]
            burst_time = original_process[2]
            
            turnaround_time = end - arrival_time
            waiting_time = start - arrival_time
            
            total_turnaround += turnaround_time
            total_waiting += waiting_time
            
            print(f"   {name}: Turnaround={turnaround_time}, Waiting={waiting_time}")
        
        avg_turnaround = total_turnaround / len(processes)
        avg_waiting = total_waiting / len(processes)
        
        print(f"\nAverage Turnaround Time: {avg_turnaround:.2f}")
        print(f"Average Waiting Time: {avg_waiting:.2f}")
        
        return schedule
    
    def load_balancer_simulation(self, servers: List[Tuple[str, int]], 
                                requests: List[Tuple[str, int]]) -> Dict[str, List[str]]:
        """
        Load Balancer using Priority Queue (Least Load First)
        
        Company: System Design, Distributed Systems
        Difficulty: Medium
        Time: O(n log k), Space: O(k) where n is requests, k is servers
        
        Args:
            servers: List of (server_name, initial_load)
            requests: List of (request_id, processing_cost)
            
        Returns: Dictionary {server_name: [assigned_requests]}
        """
        # Priority queue: (current_load, server_name)
        server_queue = [(load, name) for name, load in servers]
        heapq.heapify(server_queue)
        
        assignments = {name: [] for name, _ in servers}
        
        print("Load Balancer Simulation (Least Load First)")
        print(f"Initial servers: {servers}")
        print(f"Incoming requests: {requests}")
        print()
        
        for i, (request_id, cost) in enumerate(requests):
            print(f"Request {i+1}: {request_id} (cost: {cost})")
            
            # Get server with least load
            current_load, server_name = heapq.heappop(server_queue)
            
            # Assign request to this server
            assignments[server_name].append(request_id)
            new_load = current_load + cost
            
            print(f"   Assigned to {server_name} (load: {current_load} -> {new_load})")
            
            # Put server back with updated load
            heapq.heappush(server_queue, (new_load, server_name))
            
            # Show current server loads
            current_loads = sorted([(load, name) for load, name in server_queue])
            print(f"   Server loads: {current_loads}")
            print()
        
        print("Final assignment:")
        for server, assigned_requests in assignments.items():
            total_load = sum(cost for req_id, cost in requests if req_id in assigned_requests)
            print(f"   {server}: {assigned_requests} (total load: {total_load})")
        
        return assignments
    
    # ==========================================
    # 3. EVENT-DRIVEN SIMULATION
    # ==========================================
    
    @dataclass
    class Event:
        """Event for discrete event simulation"""
        time: float
        event_type: str
        data: Any = None
        
        def __lt__(self, other):
            return self.time < other.time
    
    def discrete_event_simulation(self, events: List[Tuple[float, str, Any]] = None) -> None:
        """
        Discrete Event Simulation using Priority Queue
        
        Company: Simulation software, Performance modeling
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Simulates a simple bank queue system
        """
        # Initialize simulation
        event_queue = []
        current_time = 0.0
        
        # Simulation state
        customers_in_system = 0
        server_busy = False
        next_customer_id = 1
        total_customers_served = 0
        total_wait_time = 0.0
        
        print("=== DISCRETE EVENT SIMULATION: Bank Queue System ===")
        print()
        
        # Schedule initial customer arrivals
        arrival_times = [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0]
        for arrival_time in arrival_times:
            event = PriorityQueueApplications.Event(arrival_time, "ARRIVAL", next_customer_id)
            heapq.heappush(event_queue, event)
            next_customer_id += 1
        
        print("Scheduled customer arrivals:", arrival_times)
        print()
        
        step = 1
        while event_queue:
            # Get next event
            event = heapq.heappop(event_queue)
            current_time = event.time
            
            print(f"Step {step}: Time {current_time:.1f} - {event.event_type}")
            
            if event.event_type == "ARRIVAL":
                customer_id = event.data
                customers_in_system += 1
                
                print(f"   Customer {customer_id} arrives")
                print(f"   Customers in system: {customers_in_system}")
                
                if not server_busy:
                    # Server available, start service immediately
                    server_busy = True
                    service_time = 2.0  # Fixed service time
                    departure_time = current_time + service_time
                    
                    departure_event = PriorityQueueApplications.Event(departure_time, "DEPARTURE", customer_id)
                    heapq.heappush(event_queue, departure_event)
                    
                    print(f"   Server available, customer {customer_id} starts service")
                    print(f"   Service will complete at time {departure_time:.1f}")
                else:
                    print(f"   Server busy, customer {customer_id} joins queue")
            
            elif event.event_type == "DEPARTURE":
                customer_id = event.data
                customers_in_system -= 1
                total_customers_served += 1
                
                print(f"   Customer {customer_id} completes service and departs")
                print(f"   Customers in system: {customers_in_system}")
                
                if customers_in_system > 0:
                    # More customers waiting, start serving next one
                    service_time = 2.0
                    departure_time = current_time + service_time
                    
                    next_customer_event = PriorityQueueApplications.Event(departure_time, "DEPARTURE", "next_customer")
                    heapq.heappush(event_queue, next_customer_event)
                    
                    print(f"   Next customer starts service, will complete at {departure_time:.1f}")
                else:
                    # No more customers, server becomes idle
                    server_busy = False
                    print(f"   Server becomes idle")
            
            print(f"   Events remaining: {len(event_queue)}")
            print()
            step += 1
        
        print("Simulation completed!")
        print(f"Total customers served: {total_customers_served}")
        print(f"Final time: {current_time:.1f}")
    
    # ==========================================
    # 4. ADVANCED OPTIMIZATION PROBLEMS
    # ==========================================
    
    def huffman_coding(self, text: str) -> Tuple[Dict[str, str], str]:
        """
        Huffman Coding using Priority Queue
        
        Company: Compression algorithms, Data storage
        Difficulty: Medium
        Time: O(n log n), Space: O(n)
        
        Returns: (character_codes, encoded_text)
        """
        from collections import Counter
        
        # Count character frequencies
        freq_count = Counter(text)
        
        print(f"Huffman Coding for text: '{text}'")
        print(f"Character frequencies: {dict(freq_count)}")
        print()
        
        # Create priority queue with character frequencies
        # Format: (frequency, unique_id, node_data)
        pq = []
        node_id = 0
        
        for char, freq in freq_count.items():
            heapq.heappush(pq, (freq, node_id, {'char': char, 'left': None, 'right': None}))
            node_id += 1
        
        print("Building Huffman tree:")
        
        # Build Huffman tree
        while len(pq) > 1:
            # Get two nodes with smallest frequencies
            freq1, id1, node1 = heapq.heappop(pq)
            freq2, id2, node2 = heapq.heappop(pq)
            
            print(f"   Merging: freq={freq1} and freq={freq2}")
            
            # Create new internal node
            merged_freq = freq1 + freq2
            merged_node = {
                'char': None,  # Internal node has no character
                'left': node1,
                'right': node2
            }
            
            heapq.heappush(pq, (merged_freq, node_id, merged_node))
            print(f"   Created internal node with frequency {merged_freq}")
            node_id += 1
        
        # Get root of Huffman tree
        if pq:
            _, _, root = pq[0]
        else:
            # Single character case
            root = {'char': text[0], 'left': None, 'right': None}
        
        # Generate codes
        def generate_codes(node, code="", codes={}):
            if node['char'] is not None:
                # Leaf node
                codes[node['char']] = code if code else "0"  # Handle single char case
            else:
                # Internal node
                if node['left']:
                    generate_codes(node['left'], code + "0", codes)
                if node['right']:
                    generate_codes(node['right'], code + "1", codes)
            return codes
        
        character_codes = generate_codes(root)
        
        print("\nGenerated Huffman codes:")
        for char, code in sorted(character_codes.items()):
            print(f"   '{char}': {code}")
        
        # Encode text
        encoded_text = ''.join(character_codes[char] for char in text)
        
        print(f"\nOriginal text: '{text}' ({len(text) * 8} bits with ASCII)")
        print(f"Encoded text: '{encoded_text}' ({len(encoded_text)} bits)")
        print(f"Compression ratio: {len(encoded_text) / (len(text) * 8):.2f}")
        
        return character_codes, encoded_text
    
    def task_scheduling_with_deadlines(self, tasks: List[Tuple[str, int, int, int]]) -> List[str]:
        """
        Task Scheduling with Deadlines using Priority Queue
        
        Company: Project management, Operating systems
        Difficulty: Hard
        Time: O(n log n), Space: O(n)
        
        Args:
            tasks: List of (task_name, processing_time, deadline, penalty)
            
        Returns: Optimal task execution order
        """
        print("Task Scheduling with Deadlines")
        print("Tasks (name, processing_time, deadline, penalty):")
        for name, proc_time, deadline, penalty in tasks:
            print(f"   {name}: process={proc_time}, deadline={deadline}, penalty={penalty}")
        print()
        
        # Strategy: Schedule tasks by deadline, then by penalty/time ratio
        scheduled_tasks = []
        remaining_tasks = list(tasks)
        current_time = 0
        
        while remaining_tasks:
            # Create priority queue for remaining tasks
            # Priority: (deadline, -penalty_per_time, processing_time, task_name)
            pq = []
            
            for name, proc_time, deadline, penalty in remaining_tasks:
                penalty_per_time = penalty / proc_time if proc_time > 0 else float('inf')
                priority = (deadline, -penalty_per_time, proc_time, name)
                heapq.heappush(pq, priority)
            
            # Select best task to schedule next
            deadline, neg_penalty_per_time, proc_time, task_name = heapq.heappop(pq)
            
            # Find full task info
            selected_task = next(task for task in remaining_tasks if task[0] == task_name)
            name, proc_time, deadline, penalty = selected_task
            
            # Schedule the task
            start_time = current_time
            end_time = current_time + proc_time
            
            print(f"Time {start_time}-{end_time}: Schedule {name}")
            print(f"   Deadline: {deadline}, ", end="")
            
            if end_time <= deadline:
                print("✓ On time")
            else:
                print(f"✗ Late by {end_time - deadline}, penalty: {penalty}")
            
            scheduled_tasks.append(name)
            remaining_tasks.remove(selected_task)
            current_time = end_time
            print()
        
        print(f"Final schedule: {' -> '.join(scheduled_tasks)}")
        
        # Calculate total penalty
        current_time = 0
        total_penalty = 0
        
        for task_name in scheduled_tasks:
            task_info = next(task for task in tasks if task[0] == task_name)
            name, proc_time, deadline, penalty = task_info
            
            current_time += proc_time
            if current_time > deadline:
                total_penalty += penalty
        
        print(f"Total penalty: {total_penalty}")
        
        return scheduled_tasks


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_priority_queue_applications():
    """Demonstrate all priority queue applications"""
    print("=== PRIORITY QUEUE APPLICATIONS DEMONSTRATION ===\n")
    
    apps = PriorityQueueApplications()
    
    # 1. Graph algorithms
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
    print("\n" + "-"*60 + "\n")
    
    print("2. Prim's Minimum Spanning Tree:")
    mst_graph = {
        'A': [('B', 2), ('C', 3)],
        'B': [('A', 2), ('C', 1), ('D', 1)],
        'C': [('A', 3), ('B', 1), ('D', 4)],
        'D': [('B', 1), ('C', 4)]
    }
    apps.prim_minimum_spanning_tree(mst_graph)
    print("\n" + "-"*60 + "\n")
    
    print("3. A* Pathfinding:")
    grid = [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    apps.a_star_pathfinding(grid, (0, 0), (4, 4))
    print("\n" + "="*60 + "\n")
    
    # 2. Scheduling applications
    print("=== SCHEDULING APPLICATIONS ===")
    
    print("1. CPU Process Scheduling:")
    processes = [
        ("P1", 0, 6, 3),   # (name, arrival, burst, priority)
        ("P2", 1, 8, 1),
        ("P3", 2, 7, 4),
        ("P4", 3, 3, 2)
    ]
    apps.cpu_process_scheduling(processes)
    print("\n" + "-"*60 + "\n")
    
    print("2. Load Balancer Simulation:")
    servers = [("Server1", 0), ("Server2", 0), ("Server3", 0)]
    requests = [("Req1", 3), ("Req2", 1), ("Req3", 4), ("Req4", 2), ("Req5", 5)]
    apps.load_balancer_simulation(servers, requests)
    print("\n" + "="*60 + "\n")
    
    # 3. Event simulation
    print("=== EVENT-DRIVEN SIMULATION ===")
    apps.discrete_event_simulation()
    print("\n" + "="*60 + "\n")
    
    # 4. Advanced optimization
    print("=== ADVANCED OPTIMIZATION ===")
    
    print("1. Huffman Coding:")
    apps.huffman_coding("ABRACADABRA")
    print("\n" + "-"*60 + "\n")
    
    print("2. Task Scheduling with Deadlines:")
    tasks = [
        ("Task1", 3, 6, 10),   # (name, processing_time, deadline, penalty)
        ("Task2", 2, 4, 5),
        ("Task3", 4, 8, 15),
        ("Task4", 1, 3, 8)
    ]
    apps.task_scheduling_with_deadlines(tasks)


if __name__ == "__main__":
    demonstrate_priority_queue_applications()
    
    print("\n=== PRIORITY QUEUE APPLICATIONS MASTERY GUIDE ===")
    
    print("\n🎯 APPLICATION CATEGORIES:")
    print("• Graph Algorithms: Shortest paths, MST, pathfinding")
    print("• Scheduling: CPU processes, task management, load balancing")
    print("• Simulation: Event-driven systems, queuing models")
    print("• Optimization: Huffman coding, resource allocation")
    print("• AI/Games: A* search, decision trees, planning")
    
    print("\n📊 ALGORITHM COMPLEXITIES:")
    print("• Dijkstra's Algorithm: O((V + E) log V)")
    print("• Prim's MST: O(E log V)")
    print("• A* Search: O(b^d) where b=branching factor, d=depth")
    print("• Process Scheduling: O(n log n)")
    print("• Huffman Coding: O(n log n)")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use appropriate priority function for problem domain")
    print("• Consider tie-breaking rules for equal priorities")
    print("• Implement early termination when possible")
    print("• Cache expensive priority calculations")
    print("• Use decrease-key operations when available")
    
    print("\n🔧 IMPLEMENTATION CONSIDERATIONS:")
    print("• Choose min vs max heap based on problem requirements")
    print("• Handle floating-point priorities carefully")
    print("• Implement stable sorting for deterministic results")
    print("• Consider memory usage for large-scale problems")
    print("• Add debugging/visualization for complex algorithms")
    
    print("\n🏆 REAL-WORLD IMPACT:")
    print("• Navigation Systems: GPS routing, traffic optimization")
    print("• Operating Systems: Process and resource scheduling")
    print("• Network Systems: Packet routing, load balancing")
    print("• Game Development: AI pathfinding, behavior trees")
    print("• Compression: Data encoding, bandwidth optimization")
    print("• Simulation: Modeling complex systems, predictions")
    
    print("\n🎓 ADVANCED TOPICS TO EXPLORE:")
    print("• Fibonacci heaps for improved decrease-key operations")
    print("• Parallel priority queue algorithms")
    print("• Persistent priority queues for functional programming")
    print("• Approximate priority queues for high-performance systems")
    print("• Multi-level priority queues for complex scheduling")
    
    print("\n📚 FURTHER APPLICATIONS:")
    print("• Machine Learning: Best-first search, beam search")
    print("• Database Systems: Query optimization, index maintenance")
    print("• Distributed Systems: Event ordering, consensus algorithms")
    print("• Financial Systems: Order matching, risk management")
    print("• Scientific Computing: Numerical optimization, simulation")
