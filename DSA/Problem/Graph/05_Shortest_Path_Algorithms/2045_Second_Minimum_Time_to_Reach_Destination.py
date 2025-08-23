"""
2045. Second Minimum Time to Reach Destination
Difficulty: Hard

Problem:
A city consists of n intersections numbered from 1 to n with bi-directional roads between 
some intersections. The inputs are generated such that you can reach any intersection from 
any other intersection and that there is exactly one road between any two directly connected 
intersections.

You are given an integer n and a 2D integer array edges where edges[i] = [ui, vi] represents 
a bi-directional road between intersections ui and vi. You are also given integers time, change.

The time needed to traverse any road is time minutes. At every intersection, there is a 
traffic light which switches its color from green to red and vice versa every change minutes. 
All traffic lights turn red at time 0. You cannot enter an intersection when the light is red.

You can leave an intersection when the light turns green, or if you are already at the 
intersection when the light turns green.

Return the second minimum time to reach intersection n from intersection 1.

Examples:
Input: n = 5, edges = [[1,2],[1,3],[1,4],[3,4],[4,5]], time = 3, change = 5
Output: 13

Input: n = 2, edges = [[1,2]], time = 3, change = 2
Output: 11

Constraints:
- 2 <= n <= 10^4
- n - 1 <= edges.length <= min(2 * 10^4, n * (n - 1) / 2)
- edges[i].length == 2
- 1 <= ui, vi <= n
- ui != vi
- There are no duplicate edges
- Each intersection can be reached from any other intersection
- 1 <= time, change <= 10^3
"""

from typing import List
import heapq
from collections import defaultdict, deque

class Solution:
    def secondMinimum_approach1_dijkstra_k_shortest(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        """
        Approach 1: Modified Dijkstra for K-th Shortest Path (Optimal)
        
        Track first and second shortest times to each node with traffic light logic.
        
        Time: O(E log V)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def get_actual_time(arrival_time):
            """Calculate actual time considering traffic lights"""
            # Traffic lights: green at [0, change), red at [change, 2*change), etc.
            cycle_time = arrival_time % (2 * change)
            
            if cycle_time < change:
                # Green light, can proceed immediately
                return arrival_time
            else:
                # Red light, wait until next green
                wait_time = 2 * change - cycle_time
                return arrival_time + wait_time
        
        # Track first and second shortest times to each node
        first_time = [float('inf')] * (n + 1)
        second_time = [float('inf')] * (n + 1)
        
        # Priority queue: (time, node)
        pq = [(0, 1)]
        first_time[1] = 0
        
        while pq:
            current_time, node = heapq.heappop(pq)
            
            # If we've found second shortest to destination
            if node == n and second_time[n] != float('inf'):
                return second_time[n]
            
            # Skip if this time is worse than second shortest
            if current_time > second_time[node]:
                continue
            
            # Calculate when we can actually leave this intersection
            departure_time = get_actual_time(current_time)
            
            for neighbor in graph[node]:
                arrival_time = departure_time + time
                
                if arrival_time < first_time[neighbor]:
                    # Found new shortest path
                    second_time[neighbor] = first_time[neighbor]
                    first_time[neighbor] = arrival_time
                    heapq.heappush(pq, (arrival_time, neighbor))
                elif arrival_time < second_time[neighbor] and arrival_time != first_time[neighbor]:
                    # Found new second shortest path (different from first)
                    second_time[neighbor] = arrival_time
                    heapq.heappush(pq, (arrival_time, neighbor))
        
        return second_time[n]
    
    def secondMinimum_approach2_bfs_level_tracking(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        """
        Approach 2: BFS with Level Tracking
        
        Use BFS to find shortest and second shortest paths considering traffic lights.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def get_next_departure_time(arrival_time):
            """Get next valid departure time considering traffic lights"""
            cycle_time = arrival_time % (2 * change)
            
            if cycle_time < change:
                # Green light
                return arrival_time
            else:
                # Red light, wait for next green
                return arrival_time + (2 * change - cycle_time)
        
        # Track times to reach each node
        times = [[] for _ in range(n + 1)]
        times[1] = [0]
        
        queue = deque([(1, 0)])  # (node, time)
        
        while queue:
            node, current_time = queue.popleft()
            
            # Calculate departure time
            departure_time = get_next_departure_time(current_time)
            
            for neighbor in graph[node]:
                arrival_time = departure_time + time
                
                # Only keep first two shortest times
                if len(times[neighbor]) == 0:
                    times[neighbor].append(arrival_time)
                    queue.append((neighbor, arrival_time))
                elif len(times[neighbor]) == 1 and arrival_time != times[neighbor][0]:
                    times[neighbor].append(arrival_time)
                    queue.append((neighbor, arrival_time))
                    
                    # If this is destination and we have second time
                    if neighbor == n:
                        return times[n][1]
        
        return times[n][1] if len(times[n]) > 1 else -1
    
    def secondMinimum_approach3_shortest_path_tree(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        """
        Approach 3: Build Shortest Path Tree and Find Alternative
        
        Build shortest path tree, then find second shortest by exploring alternatives.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Find shortest path distances (in number of edges)
        distances = [-1] * (n + 1)
        distances[1] = 0
        queue = deque([1])
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        shortest_path_length = distances[n]
        
        def calculate_time_for_path_length(path_length):
            """Calculate actual time for given path length considering traffic lights"""
            total_time = 0
            
            for step in range(path_length):
                # Check if we need to wait at traffic light
                cycle_time = total_time % (2 * change)
                
                if cycle_time >= change:
                    # Red light, wait for green
                    total_time += (2 * change - cycle_time)
                
                # Travel time
                total_time += time
            
            return total_time
        
        # Calculate time for shortest path
        shortest_time = calculate_time_for_path_length(shortest_path_length)
        
        # Second shortest path is either:
        # 1. A path of length shortest_path_length + 1 (if exists)
        # 2. A path of length shortest_path_length + 2
        
        # Check if there's a path of length shortest_path_length + 1
        has_alternative_shortest = False
        
        # Use BFS to check for alternative paths
        queue = deque([(1, 0)])
        visited = set()
        
        while queue:
            node, dist = queue.popleft()
            
            if (node, dist) in visited:
                continue
            visited.add((node, dist))
            
            if node == n and dist == shortest_path_length + 1:
                has_alternative_shortest = True
                break
            
            if dist <= shortest_path_length + 1:
                for neighbor in graph[node]:
                    queue.append((neighbor, dist + 1))
        
        if has_alternative_shortest:
            return calculate_time_for_path_length(shortest_path_length + 1)
        else:
            return calculate_time_for_path_length(shortest_path_length + 2)
    
    def secondMinimum_approach4_state_space_search(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        """
        Approach 4: State Space Search with Time Tracking
        
        Model as state space where state includes current node and current time.
        
        Time: O(V * T) where T is maximum time considered
        Space: O(V * T)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def is_green_light(current_time):
            """Check if traffic light is green at given time"""
            return (current_time // change) % 2 == 0
        
        # BFS in state space: (node, time, is_second_visit)
        visited = set()
        queue = deque([(1, 0, False)])  # Start at node 1, time 0, first visit
        
        while queue:
            node, current_time, is_second = queue.popleft()
            
            # If reached destination for second time
            if node == n and is_second:
                return current_time
            
            # State: (node, time modulo 2*change, is_second)
            state = (node, current_time % (2 * change), is_second)
            if state in visited:
                continue
            visited.add(state)
            
            # Wait if red light
            if not is_green_light(current_time):
                next_green = ((current_time // change) + 1) * change
                queue.append((node, next_green, is_second))
            else:
                # Can move to neighbors
                for neighbor in graph[node]:
                    arrival_time = current_time + time
                    
                    # Check if this is second visit to destination
                    new_is_second = is_second or (neighbor == n and node != n)
                    
                    queue.append((neighbor, arrival_time, new_is_second))
        
        return -1
    
    def secondMinimum_approach5_path_enumeration(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        """
        Approach 5: Explicit Path Enumeration (Educational)
        
        Find all shortest and near-shortest paths, calculate times.
        
        Time: O(E^k) where k is path length
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def calculate_path_time(path_edges):
            """Calculate time for given path considering traffic lights"""
            total_time = 0
            
            for _ in range(path_edges):
                # Check traffic light
                cycle_time = total_time % (2 * change)
                
                if cycle_time >= change:
                    # Red light, wait
                    total_time += (2 * change - cycle_time)
                
                # Travel
                total_time += time
            
            return total_time
        
        # Find shortest path length using BFS
        distances = [-1] * (n + 1)
        distances[1] = 0
        queue = deque([1])
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        shortest_length = distances[n]
        
        # Calculate times for possible path lengths
        times = []
        for length in [shortest_length, shortest_length + 1, shortest_length + 2]:
            path_time = calculate_path_time(length)
            times.append(path_time)
        
        # Return second distinct time
        times = list(set(times))
        times.sort()
        
        return times[1] if len(times) > 1 else times[0]

def test_second_minimum_time():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, time, change, expected)
        (5, [[1,2],[1,3],[1,4],[3,4],[4,5]], 3, 5, 13),
        (2, [[1,2]], 3, 2, 11),
        (3, [[1,2],[2,3]], 1, 2, 5),
        (4, [[1,2],[1,3],[2,4],[3,4]], 2, 3, 8),
    ]
    
    approaches = [
        ("Dijkstra K-Shortest", solution.secondMinimum_approach1_dijkstra_k_shortest),
        ("BFS Level Tracking", solution.secondMinimum_approach2_bfs_level_tracking),
        ("Shortest Path Tree", solution.secondMinimum_approach3_shortest_path_tree),
        ("State Space Search", solution.secondMinimum_approach4_state_space_search),
        ("Path Enumeration", solution.secondMinimum_approach5_path_enumeration),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, time, change, expected) in enumerate(test_cases):
            result = func(n, edges[:], time, change)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_traffic_light_logic():
    """Demonstrate traffic light timing logic"""
    print("\n=== Traffic Light Logic Demo ===")
    
    change = 5
    time = 3
    
    print(f"Traffic light cycle: Green for {change} minutes, Red for {change} minutes")
    print(f"Travel time between intersections: {time} minutes")
    
    print(f"\nTraffic light schedule:")
    for t in range(20):
        cycle_time = t % (2 * change)
        is_green = cycle_time < change
        color = "GREEN" if is_green else "RED"
        print(f"  Time {t:2d}: {color}")
    
    print(f"\nDeparture time calculation examples:")
    arrival_times = [0, 2, 5, 7, 10, 12]
    
    for arrival in arrival_times:
        cycle_time = arrival % (2 * change)
        
        if cycle_time < change:
            departure = arrival
            wait = 0
        else:
            wait = 2 * change - cycle_time
            departure = arrival + wait
        
        print(f"  Arrive at time {arrival}: wait {wait} minutes, depart at time {departure}")

def demonstrate_k_shortest_path_tracking():
    """Demonstrate K-shortest path tracking"""
    print("\n=== K-Shortest Path Tracking Demo ===")
    
    print("Tracking First and Second Shortest Times:")
    
    # Simulate tracking for a simple example
    n = 3
    first_time = [float('inf')] * (n + 1)
    second_time = [float('inf')] * (n + 1)
    
    first_time[1] = 0
    
    print(f"Initial state:")
    for i in range(1, n + 1):
        f = first_time[i] if first_time[i] != float('inf') else '∞'
        s = second_time[i] if second_time[i] != float('inf') else '∞'
        print(f"  Node {i}: first={f}, second={s}")
    
    # Simulate updates
    updates = [
        (2, 5, "First path to node 2"),
        (3, 8, "First path to node 3"),
        (2, 7, "Second path to node 2"),
        (3, 10, "Second path to node 3"),
        (3, 9, "Better second path to node 3"),
    ]
    
    for node, new_time, description in updates:
        print(f"\n{description}: time {new_time}")
        
        if new_time < first_time[node]:
            # New shortest path
            second_time[node] = first_time[node]
            first_time[node] = new_time
            print(f"  Updated: first={new_time}, second={second_time[node]}")
        elif new_time < second_time[node] and new_time != first_time[node]:
            # New second shortest path
            second_time[node] = new_time
            print(f"  Updated: first={first_time[node]}, second={new_time}")
        else:
            print(f"  No update needed")

def analyze_k_shortest_path_algorithms():
    """Analyze K-shortest path algorithms"""
    print("\n=== K-Shortest Path Algorithms Analysis ===")
    
    print("K-Shortest Path Problem Variants:")
    
    print("\n1. **Simple K-Shortest Paths:**")
    print("   • Find K shortest paths from source to destination")
    print("   • Paths may share edges and nodes")
    print("   • Yen's algorithm, Eppstein's algorithm")
    
    print("\n2. **Node-Disjoint K-Shortest Paths:**")
    print("   • K paths that don't share intermediate nodes")
    print("   • More restrictive constraint")
    print("   • Used for fault-tolerant routing")
    
    print("\n3. **Edge-Disjoint K-Shortest Paths:**")
    print("   • K paths that don't share edges")
    print("   • Less restrictive than node-disjoint")
    print("   • Network flow based algorithms")
    
    print("\n4. **K-Shortest Simple Paths:**")
    print("   • K shortest paths without cycles")
    print("   • Most common practical variant")
    print("   • This problem's approach")
    
    print("\nModified Dijkstra for K-Shortest:")
    print("• Track K shortest distances to each node")
    print("• Use priority queue with multiple entries per node")
    print("• Update rule: maintain sorted list of K best distances")
    print("• Termination: when K-th shortest to destination found")
    
    print("\nComplexity Analysis:")
    print("• Time: O(K * E log(K * V))")
    print("• Space: O(K * V)")
    print("• For K=2: O(E log V) - same as standard Dijkstra")

def analyze_traffic_light_constraints():
    """Analyze traffic light constraint modeling"""
    print("\n=== Traffic Light Constraints Analysis ===")
    
    print("Traffic Light Modeling:")
    
    print("\n1. **Periodic Constraint:**")
    print("   • Lights cycle every 2*change minutes")
    print("   • Green: [0, change), Red: [change, 2*change)")
    print("   • Pattern repeats indefinitely")
    
    print("\n2. **State Space Impact:**")
    print("   • Standard: state = current_node")
    print("   • With traffic lights: state = (current_node, time_mod_cycle)")
    print("   • Increases state space by factor of 2*change")
    
    print("\n3. **Time Calculation:**")
    print("   • Arrival time at intersection")
    print("   • Wait time if red light")
    print("   • Departure time when green")
    print("   • Travel time to next intersection")
    
    print("\n4. **Optimization Strategies:**")
    print("   • Early termination when second shortest found")
    print("   • State pruning based on time modulo cycle")
    print("   • Bidirectional search from start and end")
    
    print("\nReal-World Considerations:")
    print("• **Traffic synchronization:** Coordinated light timing")
    print("• **Dynamic timing:** Lights adapt to traffic flow")
    print("• **Priority vehicles:** Emergency vehicle overrides")
    print("• **Pedestrian phases:** Additional timing constraints")
    print("• **Rush hour patterns:** Time-dependent cycles")

def compare_second_shortest_applications():
    """Compare applications of second shortest path algorithms"""
    print("\n=== Second Shortest Path Applications ===")
    
    print("1. **Network Reliability:**")
    print("   • Primary and backup routing paths")
    print("   • Fault tolerance in communication networks")
    print("   • Redundant system design")
    
    print("\n2. **Transportation Planning:**")
    print("   • Alternative routes for navigation")
    print("   • Emergency evacuation planning")
    print("   • Load balancing on road networks")
    
    print("\n3. **Supply Chain Management:**")
    print("   • Primary and secondary supplier routes")
    print("   • Risk mitigation strategies")
    print("   • Contingency planning")
    
    print("\n4. **Game AI and Pathfinding:**")
    print("   • NPC behavior variation")
    print("   • Player choice alternatives")
    print("   • Dynamic difficulty adjustment")
    
    print("\n5. **Financial Systems:**")
    print("   • Investment strategy alternatives")
    print("   • Risk-adjusted portfolio optimization")
    print("   • Market making strategies")
    
    print("\n6. **Project Management:**")
    print("   • Critical path alternatives")
    print("   • Resource allocation options")
    print("   • Schedule optimization")
    
    print("\nKey Benefits:")
    print("• **Robustness:** Backup options when primary fails")
    print("• **Load distribution:** Spread traffic across alternatives")
    print("• **User choice:** Provide options to users")
    print("• **Risk management:** Reduce dependency on single path")
    print("• **Performance:** Balance load for better overall performance")

if __name__ == "__main__":
    test_second_minimum_time()
    demonstrate_traffic_light_logic()
    demonstrate_k_shortest_path_tracking()
    analyze_k_shortest_path_algorithms()
    analyze_traffic_light_constraints()
    compare_second_shortest_applications()

"""
Shortest Path Concepts:
1. K-th Shortest Path Algorithms
2. Time-Dependent Graph Constraints
3. Traffic Light and Scheduling Constraints
4. Modified Dijkstra for Multiple Solutions
5. State Space Extension for Temporal Constraints

Key Problem Insights:
- Find second shortest path with traffic light constraints
- Traffic lights create periodic waiting times
- K-shortest path requires tracking multiple distances
- Time-dependent constraints affect state space

Algorithm Strategy:
1. Track first and second shortest times to each node
2. Use modified Dijkstra with traffic light logic
3. Calculate actual departure times considering lights
4. Update first/second shortest times appropriately

Traffic Light Logic:
- Cycle: Green for 'change' minutes, Red for 'change' minutes
- Calculate departure time based on arrival time and cycle
- Wait if arriving during red light phase
- Proceed immediately if arriving during green phase

K-Shortest Path Tracking:
- Maintain first_time[] and second_time[] arrays
- Update rules for new shortest and second shortest
- Priority queue with multiple entries per node
- Early termination when second shortest to destination found

State Space Considerations:
- Standard state: current node
- Extended state: (current_node, time_modulo_cycle)
- Temporal constraints increase complexity
- Pruning strategies important for efficiency

Advanced Techniques:
- Modified Dijkstra for K shortest paths
- BFS with level tracking for unweighted graphs
- State space search with time modeling
- Path enumeration for small graphs

Real-world Applications:
- Traffic routing with signal coordination
- Network routing with maintenance windows
- Supply chain with scheduled operations
- Project scheduling with resource constraints
- Emergency planning with temporal factors

This problem demonstrates advanced shortest path
algorithms with temporal constraints and K-th solution finding.
"""
