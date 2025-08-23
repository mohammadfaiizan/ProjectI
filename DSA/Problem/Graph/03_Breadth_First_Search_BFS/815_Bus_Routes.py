"""
815. Bus Routes - Multiple Approaches
Difficulty: Hard

You are given an array routes representing bus routes where routes[i] is a bus route that the ith bus repeats forever.

For example, if routes[0] = [1, 5, 7], this means that the 0th bus travels in the sequence 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... forever.

You will start at the bus stop source and want to go to the bus stop target. You can travel between bus stops by buses only.

Return the minimum number of buses you must take to travel from source to target. Return -1 if it is not possible.
"""

from typing import List, Dict, Set
from collections import deque, defaultdict

class BusRoutes:
    """Multiple approaches to find minimum bus transfers"""
    
    def numBusesToDestination_bfs_stop_to_routes(self, routes: List[List[int]], source: int, target: int) -> int:
        """
        Approach 1: BFS with Stop-to-Routes Mapping
        
        Map each stop to routes that visit it, then BFS on routes.
        
        Time: O(N*M + N*M) where N=routes, M=avg stops per route
        Space: O(N*M)
        """
        if source == target:
            return 0
        
        # Build stop to routes mapping
        stop_to_routes = defaultdict(list)
        for route_idx, route in enumerate(routes):
            for stop in route:
                stop_to_routes[stop].append(route_idx)
        
        # BFS on routes
        visited_routes = set()
        queue = deque()
        
        # Start from all routes that contain source
        for route_idx in stop_to_routes[source]:
            queue.append((route_idx, 1))  # (route_index, bus_count)
            visited_routes.add(route_idx)
        
        while queue:
            current_route, bus_count = queue.popleft()
            
            # Check if target is in current route
            if target in routes[current_route]:
                return bus_count
            
            # Explore all routes connected to current route
            for stop in routes[current_route]:
                for next_route in stop_to_routes[stop]:
                    if next_route not in visited_routes:
                        visited_routes.add(next_route)
                        queue.append((next_route, bus_count + 1))
        
        return -1
    
    def numBusesToDestination_bfs_stops(self, routes: List[List[int]], source: int, target: int) -> int:
        """
        Approach 2: BFS on Stops with Route Tracking
        
        BFS on stops while tracking which routes we've used.
        
        Time: O(N*M^2), Space: O(N*M)
        """
        if source == target:
            return 0
        
        # Build stop to routes mapping
        stop_to_routes = defaultdict(set)
        for route_idx, route in enumerate(routes):
            for stop in route:
                stop_to_routes[stop].add(route_idx)
        
        # BFS on stops
        visited_stops = set([source])
        used_routes = set()
        queue = deque([(source, 0)])  # (stop, bus_count)
        
        while queue:
            current_stop, bus_count = queue.popleft()
            
            # Try all routes from current stop
            for route_idx in stop_to_routes[current_stop]:
                if route_idx in used_routes:
                    continue
                
                used_routes.add(route_idx)
                
                # Explore all stops in this route
                for next_stop in routes[route_idx]:
                    if next_stop == target:
                        return bus_count + 1
                    
                    if next_stop not in visited_stops:
                        visited_stops.add(next_stop)
                        queue.append((next_stop, bus_count + 1))
        
        return -1
    
    def numBusesToDestination_bidirectional_bfs(self, routes: List[List[int]], source: int, target: int) -> int:
        """
        Approach 3: Bidirectional BFS
        
        Search from both source and target simultaneously.
        
        Time: O(N*M), Space: O(N*M)
        """
        if source == target:
            return 0
        
        # Build stop to routes mapping
        stop_to_routes = defaultdict(set)
        for route_idx, route in enumerate(routes):
            for stop in route:
                stop_to_routes[stop].add(route_idx)
        
        # Initialize forward and backward search
        forward_routes = set(stop_to_routes[source])
        backward_routes = set(stop_to_routes[target])
        
        forward_visited = set(forward_routes)
        backward_visited = set(backward_routes)
        
        buses = 1
        
        while forward_routes and backward_routes:
            # Always expand the smaller set
            if len(forward_routes) > len(backward_routes):
                forward_routes, backward_routes = backward_routes, forward_routes
                forward_visited, backward_visited = backward_visited, forward_visited
            
            next_routes = set()
            
            for route_idx in forward_routes:
                # Check if we meet backward search
                if route_idx in backward_visited:
                    return buses
                
                # Expand to connected routes
                for stop in routes[route_idx]:
                    for next_route in stop_to_routes[stop]:
                        if next_route not in forward_visited:
                            forward_visited.add(next_route)
                            next_routes.add(next_route)
            
            forward_routes = next_routes
            buses += 1
        
        return -1
    
    def numBusesToDestination_optimized_graph(self, routes: List[List[int]], source: int, target: int) -> int:
        """
        Approach 4: Optimized Graph Construction
        
        Build route graph and use BFS on route connections.
        
        Time: O(N^2*M), Space: O(N^2)
        """
        if source == target:
            return 0
        
        n = len(routes)
        
        # Convert routes to sets for faster lookup
        route_sets = [set(route) for route in routes]
        
        # Build route adjacency graph
        graph = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if routes i and j share any stop
                if route_sets[i] & route_sets[j]:  # Intersection
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Find source and target routes
        source_routes = []
        target_routes = []
        
        for i, route_set in enumerate(route_sets):
            if source in route_set:
                source_routes.append(i)
            if target in route_set:
                target_routes.append(i)
        
        if not source_routes or not target_routes:
            return -1
        
        # BFS on route graph
        queue = deque()
        visited = set()
        
        for route_idx in source_routes:
            if route_idx in target_routes:
                return 1
            queue.append((route_idx, 1))
            visited.add(route_idx)
        
        while queue:
            current_route, buses = queue.popleft()
            
            for next_route in graph[current_route]:
                if next_route in target_routes:
                    return buses + 1
                
                if next_route not in visited:
                    visited.add(next_route)
                    queue.append((next_route, buses + 1))
        
        return -1
    
    def numBusesToDestination_union_find(self, routes: List[List[int]], source: int, target: int) -> int:
        """
        Approach 5: Union-Find for Route Connectivity
        
        Use Union-Find to group connected routes, then BFS.
        
        Time: O(N*M*α(N)), Space: O(N)
        """
        if source == target:
            return 0
        
        n = len(routes)
        
        # Union-Find implementation
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Convert routes to sets
        route_sets = [set(route) for route in routes]
        
        # Union routes that share stops
        for i in range(n):
            for j in range(i + 1, n):
                if route_sets[i] & route_sets[j]:
                    union(i, j)
        
        # Find source and target route groups
        source_groups = set()
        target_groups = set()
        
        for i, route_set in enumerate(route_sets):
            if source in route_set:
                source_groups.add(find(i))
            if target in route_set:
                target_groups.add(find(i))
        
        # If source and target are in same group, find minimum buses
        if source_groups & target_groups:
            # Need to do BFS within the connected component
            return self._bfs_within_component(routes, source, target, route_sets)
        
        return -1
    
    def _bfs_within_component(self, routes: List[List[int]], source: int, target: int, route_sets: List[Set[int]]) -> int:
        """Helper method for BFS within connected component"""
        stop_to_routes = defaultdict(list)
        for i, route_set in enumerate(route_sets):
            for stop in route_set:
                stop_to_routes[stop].append(i)
        
        visited_routes = set()
        queue = deque()
        
        for route_idx in stop_to_routes[source]:
            if target in route_sets[route_idx]:
                return 1
            queue.append((route_idx, 1))
            visited_routes.add(route_idx)
        
        while queue:
            current_route, buses = queue.popleft()
            
            for stop in routes[current_route]:
                for next_route in stop_to_routes[stop]:
                    if next_route not in visited_routes:
                        if target in route_sets[next_route]:
                            return buses + 1
                        visited_routes.add(next_route)
                        queue.append((next_route, buses + 1))
        
        return -1

def test_bus_routes():
    """Test bus routes algorithms"""
    solver = BusRoutes()
    
    test_cases = [
        ([[1,2,7],[3,6,7]], 1, 6, 2, "Example 1"),
        ([[7,12],[4,5,15],[6],[15,19],[9,12,13]], 15, 12, -1, "No path"),
        ([[1,2,3],[4,5,6]], 1, 6, -1, "Disconnected"),
        ([[1,7],[3,5]], 5, 5, 0, "Same source and target"),
        ([[1,2,3,4],[2,5,6,7],[5,8,9]], 1, 9, 3, "Multiple transfers"),
    ]
    
    algorithms = [
        ("BFS Stop-to-Routes", solver.numBusesToDestination_bfs_stop_to_routes),
        ("BFS on Stops", solver.numBusesToDestination_bfs_stops),
        ("Bidirectional BFS", solver.numBusesToDestination_bidirectional_bfs),
        ("Optimized Graph", solver.numBusesToDestination_optimized_graph),
    ]
    
    print("=== Testing Bus Routes ===")
    
    for routes, source, target, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Routes: {routes}, Source: {source}, Target: {target}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(routes, source, target)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Buses: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_bus_routes()

"""
Bus Routes demonstrates advanced BFS techniques for
multi-modal transportation networks with route optimization
and bidirectional search strategies.
"""
