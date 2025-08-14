"""
Greedy Graph Algorithms - MST, Shortest Path, Network Flow
==========================================================

Topics: Minimum spanning tree, shortest path algorithms, network optimization
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix
Difficulty: Medium to Hard
Time Complexity: O(E log V) for MST, O((V+E) log V) for shortest path
Space Complexity: O(V + E) for graph representation
"""

from typing import List, Tuple, Optional, Dict, Any, Set
import heapq
from collections import defaultdict, deque

class GreedyGraphAlgorithms:
    
    def __init__(self):
        """Initialize with algorithm tracking"""
        self.solution_steps = []
        self.algorithm_stats = {}
    
    # ==========================================
    # 1. MINIMUM SPANNING TREE ALGORITHMS
    # ==========================================
    
    def kruskals_mst(self, edges: List[Tuple[str, str, int]], vertices: Set[str]) -> Tuple[List[Tuple[str, str, int]], int]:
        """
        Kruskal's Minimum Spanning Tree Algorithm
        
        Company: Google, Amazon, Microsoft (network design)
        Difficulty: Hard
        Time: O(E log E), Space: O(V)
        
        Problem: Find minimum cost spanning tree
        Greedy Strategy: Add cheapest edge that doesn't create cycle
        
        Args:
            edges: List of (u, v, weight) tuples
            vertices: Set of all vertices
        
        Returns:
            (mst_edges, total_weight)
        """
        print("=== KRUSKAL'S MINIMUM SPANNING TREE ===")
        print("Problem: Find minimum cost spanning tree")
        print("Greedy Strategy: Add cheapest edge that doesn't create cycle")
        print()
        
        print(f"Vertices: {sorted(vertices)}")
        print("Edges (u, v, weight):")
        for u, v, weight in edges:
            print(f"   {u} -- {v}: {weight}")
        print()
        
        # Sort edges by weight (greedy choice)
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        print("Edges sorted by weight:")
        for u, v, weight in sorted_edges:
            print(f"   {u} -- {v}: {weight}")
        print()
        
        # Union-Find data structure for cycle detection
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(v):
            if parent[v] != v:
                parent[v] = find(parent[v])  # Path compression
            return parent[v]
        
        def union(u, v):
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                # Union by rank
                if rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                elif rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                else:
                    parent[root_v] = root_u
                    rank[root_u] += 1
                return True
            return False
        
        # Kruskal's algorithm
        mst_edges = []
        total_weight = 0
        edges_added = 0
        
        print("Kruskal's algorithm execution:")
        for i, (u, v, weight) in enumerate(sorted_edges):
            print(f"Step {i+1}: Consider edge {u} -- {v} (weight: {weight})")
            
            root_u, root_v = find(u), find(v)
            print(f"   Components: {u} in {root_u}, {v} in {root_v}")
            
            if union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight
                edges_added += 1
                print(f"   ✓ Added edge (no cycle created)")
                print(f"   MST edges so far: {[(a, b, w) for a, b, w in mst_edges]}")
                print(f"   Total weight: {total_weight}")
                
                if edges_added == len(vertices) - 1:
                    print(f"   MST complete! ({edges_added} edges for {len(vertices)} vertices)")
                    break
            else:
                print(f"   ✗ Rejected (would create cycle)")
            
            print()
        
        print("Final MST:")
        for u, v, weight in mst_edges:
            print(f"   {u} -- {v}: {weight}")
        print(f"Total MST weight: {total_weight}")
        
        return mst_edges, total_weight
    
    def prims_mst(self, graph: Dict[str, List[Tuple[str, int]]], start_vertex: str) -> Tuple[List[Tuple[str, str, int]], int]:
        """
        Prim's Minimum Spanning Tree Algorithm
        
        Company: Google, Amazon, Network optimization
        Difficulty: Hard  
        Time: O(E log V), Space: O(V + E)
        
        Problem: Find minimum cost spanning tree
        Greedy Strategy: Grow tree by adding minimum weight edge to unvisited vertex
        """
        print("=== PRIM'S MINIMUM SPANNING TREE ===")
        print("Problem: Find minimum cost spanning tree")
        print("Greedy Strategy: Grow tree by adding minimum weight edge")
        print()
        
        print(f"Starting vertex: {start_vertex}")
        print("Graph adjacency list:")
        for vertex, neighbors in graph.items():
            print(f"   {vertex}: {neighbors}")
        print()
        
        # Initialize
        visited = {start_vertex}
        mst_edges = []
        total_weight = 0
        
        # Priority queue: (weight, from_vertex, to_vertex)
        pq = []
        
        # Add all edges from start vertex
        for neighbor, weight in graph[start_vertex]:
            heapq.heappush(pq, (weight, start_vertex, neighbor))
        
        print("Initial edges from start vertex:")
        for weight, from_v, to_v in sorted(pq):
            print(f"   {from_v} -- {to_v}: {weight}")
        print()
        
        step = 1
        while pq and len(visited) < len(graph):
            weight, from_vertex, to_vertex = heapq.heappop(pq)
            
            print(f"Step {step}: Consider edge {from_vertex} -- {to_vertex} (weight: {weight})")
            
            if to_vertex in visited:
                print(f"   ✗ Rejected ({to_vertex} already in MST)")
                continue
            
            # Add edge to MST
            visited.add(to_vertex)
            mst_edges.append((from_vertex, to_vertex, weight))
            total_weight += weight
            
            print(f"   ✓ Added edge {from_vertex} -- {to_vertex}")
            print(f"   Visited vertices: {sorted(visited)}")
            print(f"   Total weight: {total_weight}")
            
            # Add new edges from newly added vertex
            new_edges = 0
            for neighbor, edge_weight in graph[to_vertex]:
                if neighbor not in visited:
                    heapq.heappush(pq, (edge_weight, to_vertex, neighbor))
                    new_edges += 1
            
            if new_edges > 0:
                print(f"   Added {new_edges} new edges to priority queue")
            
            print()
            step += 1
        
        print("Final MST (Prim's algorithm):")
        for from_v, to_v, weight in mst_edges:
            print(f"   {from_v} -- {to_v}: {weight}")
        print(f"Total MST weight: {total_weight}")
        
        return mst_edges, total_weight
    
    # ==========================================
    # 2. SHORTEST PATH ALGORITHMS
    # ==========================================
    
    def dijkstra_shortest_path(self, graph: Dict[str, List[Tuple[str, int]]], start: str, end: str = None) -> Dict[str, Tuple[int, List[str]]]:
        """
        Dijkstra's Shortest Path Algorithm
        
        Company: Google, Amazon, Uber, Maps applications
        Difficulty: Hard
        Time: O((V + E) log V), Space: O(V)
        
        Problem: Find shortest paths from source to all vertices
        Greedy Strategy: Always extend path with minimum distance
        """
        print("=== DIJKSTRA'S SHORTEST PATH ALGORITHM ===")
        print("Problem: Find shortest paths from source vertex")
        print("Greedy Strategy: Always extend path with minimum total distance")
        print()
        
        print(f"Source vertex: {start}")
        if end:
            print(f"Target vertex: {end}")
        print("Graph adjacency list:")
        for vertex, neighbors in graph.items():
            print(f"   {vertex}: {neighbors}")
        print()
        
        # Initialize distances and previous vertices
        distances = {vertex: float('inf') for vertex in graph}
        distances[start] = 0
        previous = {vertex: None for vertex in graph}
        visited = set()
        
        # Priority queue: (distance, vertex)
        pq = [(0, start)]
        
        print("Dijkstra's algorithm execution:")
        step = 1
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            print(f"Step {step}: Process vertex {current_vertex} (distance: {current_distance})")
            
            if current_vertex in visited:
                print(f"   Already processed, skipping")
                continue
            
            visited.add(current_vertex)
            
            # Early termination if target reached
            if end and current_vertex == end:
                print(f"   🎯 Reached target vertex {end}!")
                break
            
            print(f"   Exploring neighbors of {current_vertex}:")
            
            # Update distances to neighbors
            for neighbor, weight in graph[current_vertex]:
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    print(f"      To {neighbor}: current={distances[neighbor]}, new={new_distance}")
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor))
                        print(f"         ✓ Updated distance to {neighbor}")
                    else:
                        print(f"         ✗ No improvement")
            
            print(f"   Current distances: {dict(distances)}")
            print()
            step += 1
        
        # Reconstruct paths
        def get_path(target):
            if distances[target] == float('inf'):
                return []
            
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]
        
        results = {}
        for vertex in graph:
            if distances[vertex] != float('inf'):
                results[vertex] = (distances[vertex], get_path(vertex))
        
        print("Final shortest paths:")
        for vertex, (distance, path) in results.items():
            print(f"   {start} → {vertex}: distance {distance}, path {' → '.join(path)}")
        
        return results
    
    def bellman_ford_shortest_path(self, edges: List[Tuple[str, str, int]], vertices: Set[str], start: str) -> Tuple[Dict[str, int], Dict[str, str], bool]:
        """
        Bellman-Ford Shortest Path Algorithm (handles negative weights)
        
        Company: Advanced graph algorithms, Google
        Difficulty: Hard
        Time: O(VE), Space: O(V)
        
        Problem: Find shortest paths with negative edge weights
        Strategy: Relax all edges V-1 times, then check for negative cycles
        """
        print("=== BELLMAN-FORD SHORTEST PATH ALGORITHM ===")
        print("Problem: Find shortest paths (handles negative weights)")
        print("Strategy: Relax all edges V-1 times")
        print()
        
        print(f"Source vertex: {start}")
        print(f"Vertices: {sorted(vertices)}")
        print("Edges (u, v, weight):")
        for u, v, weight in edges:
            print(f"   {u} → {v}: {weight}")
        print()
        
        # Initialize distances
        distances = {v: float('inf') for v in vertices}
        distances[start] = 0
        previous = {v: None for v in vertices}
        
        print("Bellman-Ford relaxation phases:")
        
        # Relax edges V-1 times
        for phase in range(len(vertices) - 1):
            print(f"Phase {phase + 1}:")
            updated = False
            
            for u, v, weight in edges:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    old_distance = distances[v]
                    distances[v] = distances[u] + weight
                    previous[v] = u
                    updated = True
                    
                    print(f"   Relaxed edge {u} → {v}: {old_distance} → {distances[v]}")
            
            if not updated:
                print(f"   No updates in this phase - algorithm can terminate early")
                break
            else:
                print(f"   Current distances: {dict(distances)}")
            print()
        
        # Check for negative cycles
        print("Checking for negative cycles:")
        has_negative_cycle = False
        
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                print(f"   ✗ Negative cycle detected via edge {u} → {v}")
                has_negative_cycle = True
                break
        
        if not has_negative_cycle:
            print("   ✓ No negative cycles found")
        
        print("\nFinal shortest distances:")
        for vertex in sorted(vertices):
            if distances[vertex] != float('inf'):
                print(f"   {start} → {vertex}: {distances[vertex]}")
            else:
                print(f"   {start} → {vertex}: unreachable")
        
        return distances, previous, has_negative_cycle
    
    # ==========================================
    # 3. NETWORK FLOW ALGORITHMS
    # ==========================================
    
    def maximum_flow_ford_fulkerson(self, graph: Dict[str, Dict[str, int]], source: str, sink: str) -> Tuple[int, Dict[Tuple[str, str], int]]:
        """
        Maximum Flow using Ford-Fulkerson with DFS (Greedy path finding)
        
        Company: Google, Network optimization, Amazon
        Difficulty: Hard
        Time: O(Ef), Space: O(V + E) where f is max flow value
        
        Problem: Find maximum flow from source to sink
        Greedy Strategy: Find any augmenting path and push flow
        """
        print("=== MAXIMUM FLOW (FORD-FULKERSON) ===")
        print("Problem: Find maximum flow from source to sink")
        print("Greedy Strategy: Find any augmenting path and push maximum flow")
        print()
        
        print(f"Source: {source}, Sink: {sink}")
        print("Capacity graph:")
        for u, neighbors in graph.items():
            for v, capacity in neighbors.items():
                print(f"   {u} → {v}: capacity {capacity}")
        print()
        
        # Initialize residual graph
        residual = defaultdict(lambda: defaultdict(int))
        for u, neighbors in graph.items():
            for v, capacity in neighbors.items():
                residual[u][v] = capacity
        
        def dfs_find_path(source, sink, visited, path, min_capacity):
            """Find augmenting path using DFS"""
            if source == sink:
                return path, min_capacity
            
            visited.add(source)
            
            for neighbor in residual[source]:
                if neighbor not in visited and residual[source][neighbor] > 0:
                    new_min = min(min_capacity, residual[source][neighbor])
                    result = dfs_find_path(neighbor, sink, visited, path + [neighbor], new_min)
                    if result:
                        return result
            
            return None
        
        max_flow = 0
        flow_edges = defaultdict(int)
        iteration = 1
        
        print("Ford-Fulkerson algorithm execution:")
        
        while True:
            # Find augmenting path
            visited = set()
            path_result = dfs_find_path(source, sink, visited, [source], float('inf'))
            
            if not path_result:
                print(f"Iteration {iteration}: No more augmenting paths found")
                break
            
            path, flow_value = path_result
            
            print(f"Iteration {iteration}:")
            print(f"   Found augmenting path: {' → '.join(path)}")
            print(f"   Path capacity: {flow_value}")
            
            # Update residual graph and flow
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual[u][v] -= flow_value  # Forward edge
                residual[v][u] += flow_value  # Backward edge
                flow_edges[(u, v)] += flow_value
            
            max_flow += flow_value
            print(f"   Added flow: {flow_value}")
            print(f"   Total flow so far: {max_flow}")
            
            # Show current residual capacities
            print("   Updated residual capacities:")
            for u, neighbors in residual.items():
                for v, capacity in neighbors.items():
                    if capacity > 0:
                        print(f"      {u} → {v}: {capacity}")
            print()
            
            iteration += 1
        
        print(f"Maximum flow from {source} to {sink}: {max_flow}")
        print("\nFlow on edges:")
        for (u, v), flow in flow_edges.items():
            if flow > 0:
                print(f"   {u} → {v}: {flow}")
        
        return max_flow, dict(flow_edges)
    
    # ==========================================
    # 4. GREEDY GRAPH COLORING
    # ==========================================
    
    def graph_coloring_greedy(self, graph: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Greedy Graph Coloring Algorithm
        
        Company: Scheduling problems, Google
        Difficulty: Medium
        Time: O(V + E), Space: O(V)
        
        Problem: Color vertices with minimum colors (no adjacent vertices same color)
        Greedy Strategy: Color vertices in order, use smallest available color
        
        Note: This is an approximation algorithm, not optimal
        """
        print("=== GREEDY GRAPH COLORING ===")
        print("Problem: Color vertices with minimum colors")
        print("Greedy Strategy: Use smallest available color for each vertex")
        print("Note: This gives approximation, not optimal coloring")
        print()
        
        print("Graph adjacency list:")
        for vertex, neighbors in graph.items():
            print(f"   {vertex}: {neighbors}")
        print()
        
        # Initialize coloring
        coloring = {}
        available_colors = set(range(len(graph)))  # 0, 1, 2, ...
        
        print("Greedy coloring process:")
        
        # Sort vertices by degree (highest degree first) - better heuristic
        vertices_by_degree = sorted(graph.keys(), key=lambda v: len(graph[v]), reverse=True)
        
        print("Vertices sorted by degree (highest first):")
        for vertex in vertices_by_degree:
            print(f"   {vertex}: degree {len(graph[vertex])}")
        print()
        
        for i, vertex in enumerate(vertices_by_degree):
            print(f"Step {i+1}: Color vertex {vertex}")
            
            # Find colors used by neighbors
            used_colors = set()
            for neighbor in graph[vertex]:
                if neighbor in coloring:
                    used_colors.add(coloring[neighbor])
            
            print(f"   Neighbors: {graph[vertex]}")
            print(f"   Colors used by neighbors: {sorted(used_colors) if used_colors else 'none'}")
            
            # Find smallest available color
            for color in available_colors:
                if color not in used_colors:
                    coloring[vertex] = color
                    print(f"   ✓ Assigned color {color} to vertex {vertex}")
                    break
            
            print(f"   Current coloring: {dict(sorted(coloring.items()))}")
            print()
        
        # Count total colors used
        colors_used = len(set(coloring.values()))
        
        print("Final coloring:")
        for vertex in sorted(coloring.keys()):
            print(f"   Vertex {vertex}: Color {coloring[vertex]}")
        
        print(f"\nTotal colors used: {colors_used}")
        
        # Verify coloring is valid
        print("\nVerification:")
        valid = True
        for vertex, neighbors in graph.items():
            for neighbor in neighbors:
                if coloring[vertex] == coloring[neighbor]:
                    print(f"   ✗ Invalid: {vertex} and {neighbor} have same color")
                    valid = False
        
        if valid:
            print("   ✓ Valid coloring - no adjacent vertices have same color")
        
        return coloring
    
    # ==========================================
    # 5. TRAVELING SALESMAN PROBLEM (GREEDY APPROXIMATIONS)
    # ==========================================
    
    def tsp_nearest_neighbor(self, distances: Dict[Tuple[str, str], int], start_city: str) -> Tuple[List[str], int]:
        """
        Traveling Salesman Problem - Nearest Neighbor Heuristic
        
        Company: Logistics, Google, Amazon
        Difficulty: Hard (NP-hard problem)
        Time: O(n²), Space: O(n)
        
        Problem: Find shortest tour visiting all cities exactly once
        Greedy Strategy: Always go to nearest unvisited city
        
        Note: This is a 2-approximation algorithm for metric TSP
        """
        print("=== TSP: NEAREST NEIGHBOR HEURISTIC ===")
        print("Problem: Find shortest tour visiting all cities")
        print("Greedy Strategy: Always go to nearest unvisited city")
        print("Note: This gives 2-approximation for metric TSP")
        print()
        
        # Extract all cities
        cities = set()
        for (u, v), dist in distances.items():
            cities.add(u)
            cities.add(v)
        
        print(f"Cities: {sorted(cities)}")
        print(f"Starting city: {start_city}")
        print()
        print("Distance matrix:")
        
        # Print distance matrix
        sorted_cities = sorted(cities)
        print("      ", end="")
        for city in sorted_cities:
            print(f"{city:6}", end="")
        print()
        
        for city1 in sorted_cities:
            print(f"{city1:4}: ", end="")
            for city2 in sorted_cities:
                if city1 == city2:
                    print(f"{'0':6}", end="")
                else:
                    dist = distances.get((city1, city2), distances.get((city2, city1), '∞'))
                    print(f"{dist:6}", end="")
            print()
        print()
        
        # Nearest neighbor algorithm
        tour = [start_city]
        unvisited = cities - {start_city}
        total_distance = 0
        current_city = start_city
        
        print("Nearest neighbor tour construction:")
        
        while unvisited:
            nearest_city = None
            min_distance = float('inf')
            
            print(f"Current city: {current_city}")
            print(f"Unvisited cities: {sorted(unvisited)}")
            
            # Find nearest unvisited city
            for city in unvisited:
                dist = distances.get((current_city, city), distances.get((city, current_city), float('inf')))
                print(f"   Distance to {city}: {dist}")
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_city = city
            
            # Move to nearest city
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            total_distance += min_distance
            current_city = nearest_city
            
            print(f"   ✓ Go to {nearest_city} (distance: {min_distance})")
            print(f"   Tour so far: {' → '.join(tour)}")
            print(f"   Total distance: {total_distance}")
            print()
        
        # Return to start city
        return_distance = distances.get((current_city, start_city), distances.get((start_city, current_city), 0))
        tour.append(start_city)
        total_distance += return_distance
        
        print(f"Return to start: {current_city} → {start_city} (distance: {return_distance})")
        print()
        print("Final TSP tour:")
        print(f"   Tour: {' → '.join(tour)}")
        print(f"   Total distance: {total_distance}")
        
        return tour, total_distance


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_greedy_graph_algorithms():
    """Demonstrate all greedy graph algorithms"""
    print("=== GREEDY GRAPH ALGORITHMS DEMONSTRATION ===\n")
    
    algorithms = GreedyGraphAlgorithms()
    
    # 1. Minimum Spanning Tree
    print("1. MINIMUM SPANNING TREE ALGORITHMS")
    
    print("a) Kruskal's Algorithm:")
    edges = [
        ('A', 'B', 4), ('A', 'H', 8), ('B', 'C', 8), ('B', 'H', 11),
        ('C', 'D', 7), ('C', 'F', 4), ('C', 'I', 2), ('D', 'E', 9),
        ('D', 'F', 14), ('E', 'F', 10), ('F', 'G', 2), ('G', 'H', 1),
        ('G', 'I', 6), ('H', 'I', 7)
    ]
    vertices = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}
    algorithms.kruskals_mst(edges, vertices)
    print("\n" + "-"*40 + "\n")
    
    print("b) Prim's Algorithm:")
    graph = {
        'A': [('B', 4), ('H', 8)],
        'B': [('A', 4), ('C', 8), ('H', 11)],
        'C': [('B', 8), ('D', 7), ('F', 4), ('I', 2)],
        'D': [('C', 7), ('E', 9), ('F', 14)],
        'E': [('D', 9), ('F', 10)],
        'F': [('C', 4), ('D', 14), ('E', 10), ('G', 2)],
        'G': [('F', 2), ('H', 1), ('I', 6)],
        'H': [('A', 8), ('B', 11), ('G', 1), ('I', 7)],
        'I': [('C', 2), ('G', 6), ('H', 7)]
    }
    algorithms.prims_mst(graph, 'A')
    print("\n" + "="*60 + "\n")
    
    # 2. Shortest Path Algorithms
    print("2. SHORTEST PATH ALGORITHMS")
    
    print("a) Dijkstra's Algorithm:")
    shortest_path_graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': []
    }
    algorithms.dijkstra_shortest_path(shortest_path_graph, 'A', 'E')
    print("\n" + "-"*40 + "\n")
    
    print("b) Bellman-Ford Algorithm:")
    bf_edges = [
        ('A', 'B', -1), ('A', 'C', 4), ('B', 'C', 3),
        ('B', 'D', 2), ('B', 'E', 2), ('D', 'B', 1),
        ('D', 'C', 5), ('E', 'D', -3)
    ]
    bf_vertices = {'A', 'B', 'C', 'D', 'E'}
    algorithms.bellman_ford_shortest_path(bf_edges, bf_vertices, 'A')
    print("\n" + "="*60 + "\n")
    
    # 3. Maximum Flow
    print("3. MAXIMUM FLOW")
    flow_graph = {
        'S': {'A': 10, 'C': 10},
        'A': {'B': 4, 'C': 2, 'D': 8},
        'B': {'T': 10},
        'C': {'B': 9, 'D': 9},
        'D': {'B': 6, 'T': 10},
        'T': {}
    }
    algorithms.maximum_flow_ford_fulkerson(flow_graph, 'S', 'T')
    print("\n" + "="*60 + "\n")
    
    # 4. Graph Coloring
    print("4. GRAPH COLORING")
    coloring_graph = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'C', 'E'],
        'C': ['A', 'B', 'D', 'E'],
        'D': ['A', 'C', 'F'],
        'E': ['B', 'C', 'F'],
        'F': ['D', 'E']
    }
    algorithms.graph_coloring_greedy(coloring_graph)
    print("\n" + "="*60 + "\n")
    
    # 5. Traveling Salesman Problem
    print("5. TRAVELING SALESMAN PROBLEM")
    tsp_distances = {
        ('A', 'B'): 10, ('A', 'C'): 15, ('A', 'D'): 20,
        ('B', 'C'): 35, ('B', 'D'): 25, ('C', 'D'): 30
    }
    algorithms.tsp_nearest_neighbor(tsp_distances, 'A')


if __name__ == "__main__":
    demonstrate_greedy_graph_algorithms()
    
    print("\n=== GREEDY GRAPH ALGORITHMS MASTERY GUIDE ===")
    
    print("\n🎯 ALGORITHM CATEGORIES:")
    print("• Minimum Spanning Tree: Kruskal's, Prim's algorithms")
    print("• Shortest Path: Dijkstra's, Bellman-Ford algorithms")
    print("• Network Flow: Ford-Fulkerson, max flow problems")
    print("• Graph Coloring: Greedy coloring heuristics")
    print("• TSP Approximations: Nearest neighbor, 2-opt")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Kruskal's MST: O(E log E) - dominated by sorting")
    print("• Prim's MST: O(E log V) - using binary heap")
    print("• Dijkstra's: O((V + E) log V) - using binary heap")
    print("• Bellman-Ford: O(VE) - relaxation-based")
    print("• Ford-Fulkerson: O(Ef) - depends on max flow value")
    print("• Graph Coloring: O(V + E) - greedy approximation")
    
    print("\n⚡ KEY STRATEGIES:")
    print("• MST: Cut property - safe edge addition")
    print("• Shortest Path: Relaxation and optimality conditions")
    print("• Max Flow: Augmenting paths and residual graphs")
    print("• Coloring: Vertex ordering and conflict avoidance")
    print("• TSP: Local optimization and nearest neighbor")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Use Union-Find for cycle detection (Kruskal's)")
    print("• Use priority queues for efficient minimum selection")
    print("• Implement proper residual graph for flow algorithms")
    print("• Consider vertex ordering heuristics for coloring")
    print("• Handle edge cases (disconnected graphs, negative weights)")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Network Design: MST for minimum cost connectivity")
    print("• Navigation: Shortest path for route planning")
    print("• Resource Allocation: Max flow for capacity planning")
    print("• Scheduling: Graph coloring for conflict resolution")
    print("• Logistics: TSP for delivery route optimization")
    
    print("\n🎓 ADVANCED CONCEPTS:")
    print("• Approximation ratios for NP-hard problems")
    print("• Online algorithms for dynamic graphs")
    print("• Parallel algorithms for large graphs")
    print("• Advanced data structures (Fibonacci heaps)")
    print("• Specialized algorithms for planar graphs")
