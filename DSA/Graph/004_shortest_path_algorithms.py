"""
Shortest Path Algorithms
This module implements various shortest path algorithms for different types of graphs.
"""

from collections import defaultdict, deque
import heapq
import sys

class ShortestPathAlgorithms:
    
    def __init__(self, directed=True, weighted=True):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.directed = directed
        self.weighted = weighted
    
    def add_edge(self, u, v, weight=1):
        """Add an edge to the graph"""
        self.vertices.add(u)
        self.vertices.add(v)
        
        if self.weighted:
            self.graph[u].append((v, weight))
            if not self.directed:
                self.graph[v].append((u, weight))
        else:
            self.graph[u].append(v)
            if not self.directed:
                self.graph[v].append(u)
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    # ==================== DIJKSTRA'S ALGORITHM ====================
    
    def dijkstra(self, source, target=None):
        """
        Dijkstra's Algorithm for shortest paths from source to all vertices
        Works only with non-negative edge weights
        
        Time Complexity: O((V + E) log V) with binary heap
        Space Complexity: O(V)
        
        Args:
            source: Starting vertex
            target: Target vertex (optional, if specified returns path to target only)
        
        Returns:
            tuple: (distances, predecessors) or (distance, path) if target specified
        """
        # Initialize distances and predecessors
        distances = {vertex: float('inf') for vertex in self.vertices}
        predecessors = {vertex: None for vertex in self.vertices}
        distances[source] = 0
        
        # Priority queue: (distance, vertex)
        pq = [(0, source)]
        visited = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            # Skip if already processed
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            
            # Early termination if target found
            if target and current_vertex == target:
                path = self._reconstruct_path(predecessors, source, target)
                return distances[target], path
            
            # Check all neighbors
            neighbors = self.get_neighbors(current_vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, weight = neighbor_info
                else:
                    neighbor, weight = neighbor_info, 1
                
                if neighbor not in visited:
                    new_distance = current_distance + weight
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        predecessors[neighbor] = current_vertex
                        heapq.heappush(pq, (new_distance, neighbor))
        
        if target:
            if distances[target] == float('inf'):
                return float('inf'), []
            path = self._reconstruct_path(predecessors, source, target)
            return distances[target], path
        
        return distances, predecessors
    
    def dijkstra_all_pairs(self):
        """
        Run Dijkstra from every vertex to get all-pairs shortest paths
        
        Returns:
            dict: distances[u][v] = shortest distance from u to v
        """
        all_distances = {}
        
        for vertex in self.vertices:
            distances, _ = self.dijkstra(vertex)
            all_distances[vertex] = distances
        
        return all_distances
    
    # ==================== BELLMAN-FORD ALGORITHM ====================
    
    def bellman_ford(self, source):
        """
        Bellman-Ford Algorithm for shortest paths (handles negative weights)
        Can detect negative cycles
        
        Time Complexity: O(VE)
        Space Complexity: O(V)
        
        Args:
            source: Starting vertex
        
        Returns:
            tuple: (distances, predecessors, has_negative_cycle)
        """
        # Initialize distances and predecessors
        distances = {vertex: float('inf') for vertex in self.vertices}
        predecessors = {vertex: None for vertex in self.vertices}
        distances[source] = 0
        
        # Get all edges
        edges = []
        for u in self.graph:
            for neighbor_info in self.graph[u]:
                if self.weighted:
                    v, weight = neighbor_info
                else:
                    v, weight = neighbor_info, 1
                edges.append((u, v, weight))
        
        # Relax edges V-1 times
        for _ in range(len(self.vertices) - 1):
            for u, v, weight in edges:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
        
        # Check for negative cycles
        has_negative_cycle = False
        negative_cycle_vertices = set()
        
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                has_negative_cycle = True
                negative_cycle_vertices.add(v)
        
        # If negative cycle exists, mark all affected vertices
        if has_negative_cycle:
            # BFS to find all vertices affected by negative cycle
            queue = deque(negative_cycle_vertices)
            while queue:
                vertex = queue.popleft()
                for neighbor_info in self.graph[vertex]:
                    if self.weighted:
                        neighbor, _ = neighbor_info
                    else:
                        neighbor = neighbor_info
                    
                    if neighbor not in negative_cycle_vertices:
                        negative_cycle_vertices.add(neighbor)
                        queue.append(neighbor)
            
            # Set distances to negative infinity for affected vertices
            for vertex in negative_cycle_vertices:
                distances[vertex] = float('-inf')
        
        return distances, predecessors, has_negative_cycle
    
    def find_negative_cycle(self):
        """
        Find a negative cycle in the graph using Bellman-Ford
        
        Returns:
            list: Vertices forming negative cycle, or None if no negative cycle
        """
        # Run Bellman-Ford from any vertex
        if not self.vertices:
            return None
        
        source = next(iter(self.vertices))
        distances, predecessors, has_negative_cycle = self.bellman_ford(source)
        
        if not has_negative_cycle:
            return None
        
        # Find a vertex affected by negative cycle
        affected_vertex = None
        for vertex in self.vertices:
            if distances[vertex] == float('-inf'):
                affected_vertex = vertex
                break
        
        if not affected_vertex:
            return None
        
        # Trace back to find the cycle
        current = affected_vertex
        for _ in range(len(self.vertices)):
            current = predecessors[current]
            if current is None:
                return None
        
        # Now current is definitely in the negative cycle
        cycle = []
        start = current
        while True:
            cycle.append(current)
            current = predecessors[current]
            if current == start:
                break
        
        return cycle[::-1]
    
    # ==================== FLOYD-WARSHALL ALGORITHM ====================
    
    def floyd_warshall(self):
        """
        Floyd-Warshall Algorithm for all-pairs shortest paths
        Can handle negative weights but not negative cycles
        
        Time Complexity: O(V³)
        Space Complexity: O(V²)
        
        Returns:
            tuple: (distances, next_vertex) matrices
        """
        vertices = list(self.vertices)
        n = len(vertices)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        
        # Initialize distance matrix
        distances = [[float('inf')] * n for _ in range(n)]
        next_vertex = [[None] * n for _ in range(n)]
        
        # Distance from vertex to itself is 0
        for i in range(n):
            distances[i][i] = 0
        
        # Fill initial distances from edges
        for u in self.graph:
            u_idx = vertex_to_index[u]
            for neighbor_info in self.graph[u]:
                if self.weighted:
                    v, weight = neighbor_info
                else:
                    v, weight = neighbor_info, 1
                v_idx = vertex_to_index[v]
                distances[u_idx][v_idx] = weight
                next_vertex[u_idx][v_idx] = v_idx
        
        # Floyd-Warshall main algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if distances[i][k] + distances[k][j] < distances[i][j]:
                        distances[i][j] = distances[i][k] + distances[k][j]
                        next_vertex[i][j] = next_vertex[i][k]
        
        # Convert back to vertex names
        result_distances = {}
        result_next = {}
        
        for i, u in enumerate(vertices):
            result_distances[u] = {}
            result_next[u] = {}
            for j, v in enumerate(vertices):
                result_distances[u][v] = distances[i][j]
                if next_vertex[i][j] is not None:
                    result_next[u][v] = vertices[next_vertex[i][j]]
                else:
                    result_next[u][v] = None
        
        return result_distances, result_next
    
    def floyd_warshall_path(self, source, target, next_matrix):
        """
        Reconstruct path using Floyd-Warshall next matrix
        
        Args:
            source: Starting vertex
            target: Ending vertex
            next_matrix: Next vertex matrix from floyd_warshall()
        
        Returns:
            list: Path from source to target
        """
        if next_matrix[source][target] is None:
            return []
        
        path = [source]
        current = source
        
        while current != target:
            current = next_matrix[current][target]
            path.append(current)
        
        return path
    
    # ==================== 0-1 BFS ====================
    
    def bfs_01(self, source, target=None):
        """
        0-1 BFS for shortest paths when edge weights are only 0 or 1
        Uses deque for O(V + E) time complexity
        
        Args:
            source: Starting vertex
            target: Target vertex (optional)
        
        Returns:
            tuple: (distances, predecessors) or (distance, path) if target specified
        """
        distances = {vertex: float('inf') for vertex in self.vertices}
        predecessors = {vertex: None for vertex in self.vertices}
        distances[source] = 0
        
        dq = deque([source])
        
        while dq:
            current = dq.popleft()
            
            if target and current == target:
                path = self._reconstruct_path(predecessors, source, target)
                return distances[target], path
            
            neighbors = self.get_neighbors(current)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, weight = neighbor_info
                    if weight not in [0, 1]:
                        raise ValueError("0-1 BFS requires edge weights to be 0 or 1")
                else:
                    neighbor, weight = neighbor_info, 1
                
                new_distance = distances[current] + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current
                    
                    if weight == 0:
                        dq.appendleft(neighbor)  # Add to front for 0-weight edges
                    else:
                        dq.append(neighbor)      # Add to back for 1-weight edges
        
        if target:
            if distances[target] == float('inf'):
                return float('inf'), []
            path = self._reconstruct_path(predecessors, source, target)
            return distances[target], path
        
        return distances, predecessors
    
    # ==================== MULTI-SOURCE BFS ====================
    
    def multi_source_bfs(self, sources):
        """
        Multi-source BFS to find shortest distance from any source to all vertices
        Useful for problems like "nearest exit" or "fire spreading"
        
        Args:
            sources: List of source vertices
        
        Returns:
            dict: distances[vertex] = shortest distance from any source to vertex
        """
        distances = {vertex: float('inf') for vertex in self.vertices}
        queue = deque()
        
        # Initialize all sources with distance 0
        for source in sources:
            if source in self.vertices:
                distances[source] = 0
                queue.append(source)
        
        while queue:
            current = queue.popleft()
            
            neighbors = self.get_neighbors(current)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, weight = neighbor_info
                else:
                    neighbor, weight = neighbor_info, 1
                
                new_distance = distances[current] + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    queue.append(neighbor)
        
        return distances
    
    # ==================== UTILITY METHODS ====================
    
    def _reconstruct_path(self, predecessors, source, target):
        """Reconstruct path from predecessors dictionary"""
        if predecessors[target] is None and source != target:
            return []
        
        path = []
        current = target
        
        while current is not None:
            path.append(current)
            current = predecessors[current]
        
        return path[::-1]
    
    def has_negative_weights(self):
        """Check if graph has negative edge weights"""
        for u in self.graph:
            for neighbor_info in self.graph[u]:
                if self.weighted:
                    _, weight = neighbor_info
                    if weight < 0:
                        return True
        return False
    
    def display(self):
        """Display the graph"""
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")
    
    def get_graph_info(self):
        """Get basic information about the graph"""
        num_vertices = len(self.vertices)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values())
        if not self.directed:
            num_edges //= 2
        
        return {
            "vertices": num_vertices,
            "edges": num_edges,
            "directed": self.directed,
            "weighted": self.weighted,
            "has_negative_weights": self.has_negative_weights()
        }


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Shortest Path Algorithms Demo ===\n")
    
    # Example 1: Dijkstra's Algorithm
    print("1. Dijkstra's Algorithm Example:")
    dijkstra_graph = ShortestPathAlgorithms(directed=True, weighted=True)
    
    # Create a weighted directed graph
    edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2)
    ]
    
    for u, v, w in edges:
        dijkstra_graph.add_edge(u, v, w)
    
    print("Graph:")
    dijkstra_graph.display()
    print(f"Graph Info: {dijkstra_graph.get_graph_info()}")
    print()
    
    # Single source shortest paths
    distances, predecessors = dijkstra_graph.dijkstra('A')
    print("Shortest distances from A:", distances)
    
    # Single source-target path
    distance, path = dijkstra_graph.dijkstra('A', 'E')
    print(f"Shortest path from A to E: {path}, Distance: {distance}")
    print()
    
    # Example 2: Bellman-Ford Algorithm
    print("2. Bellman-Ford Algorithm Example:")
    bf_graph = ShortestPathAlgorithms(directed=True, weighted=True)
    
    # Graph with negative edges
    bf_edges = [
        ('S', 'A', 5), ('S', 'B', 4),
        ('A', 'B', -3), ('A', 'C', 3),
        ('B', 'C', 4), ('C', 'D', 2)
    ]
    
    for u, v, w in bf_edges:
        bf_graph.add_edge(u, v, w)
    
    print("Graph with negative weights:")
    bf_graph.display()
    
    distances, predecessors, has_negative_cycle = bf_graph.bellman_ford('S')
    print(f"Shortest distances from S: {distances}")
    print(f"Has negative cycle: {has_negative_cycle}")
    print()
    
    # Example 3: Graph with negative cycle
    print("3. Negative Cycle Detection:")
    negative_cycle_graph = ShortestPathAlgorithms(directed=True, weighted=True)
    
    nc_edges = [
        ('A', 'B', 1), ('B', 'C', -3),
        ('C', 'D', 2), ('D', 'B', -1)
    ]
    
    for u, v, w in nc_edges:
        negative_cycle_graph.add_edge(u, v, w)
    
    print("Graph with negative cycle:")
    negative_cycle_graph.display()
    
    distances, _, has_negative_cycle = negative_cycle_graph.bellman_ford('A')
    print(f"Has negative cycle: {has_negative_cycle}")
    negative_cycle = negative_cycle_graph.find_negative_cycle()
    print(f"Negative cycle: {negative_cycle}")
    print()
    
    # Example 4: Floyd-Warshall Algorithm
    print("4. Floyd-Warshall Algorithm Example:")
    fw_graph = ShortestPathAlgorithms(directed=True, weighted=True)
    
    fw_edges = [
        (1, 2, 3), (1, 4, 7),
        (2, 3, 1), (3, 1, 2),
        (3, 4, 1), (4, 2, 2)
    ]
    
    for u, v, w in fw_edges:
        fw_graph.add_edge(u, v, w)
    
    print("Graph for all-pairs shortest paths:")
    fw_graph.display()
    
    all_distances, next_matrix = fw_graph.floyd_warshall()
    print("All-pairs shortest distances:")
    for u in sorted(all_distances.keys()):
        for v in sorted(all_distances[u].keys()):
            if all_distances[u][v] != float('inf'):
                print(f"  {u} -> {v}: {all_distances[u][v]}")
    
    # Show a specific path
    path = fw_graph.floyd_warshall_path(1, 4, next_matrix)
    print(f"Path from 1 to 4: {path}")
    print()
    
    # Example 5: 0-1 BFS
    print("5. 0-1 BFS Example:")
    bfs01_graph = ShortestPathAlgorithms(directed=False, weighted=True)
    
    # Graph with only 0 and 1 weights
    bfs01_edges = [
        ('A', 'B', 1), ('A', 'C', 0),
        ('B', 'D', 1), ('C', 'D', 1),
        ('C', 'E', 0), ('D', 'F', 0),
        ('E', 'F', 1)
    ]
    
    for u, v, w in bfs01_edges:
        bfs01_graph.add_edge(u, v, w)
    
    print("Graph with 0-1 weights:")
    bfs01_graph.display()
    
    distance, path = bfs01_graph.bfs_01('A', 'F')
    print(f"Shortest path from A to F: {path}, Distance: {distance}")
    print()
    
    # Example 6: Multi-source BFS
    print("6. Multi-source BFS Example:")
    multi_bfs_graph = ShortestPathAlgorithms(directed=False, weighted=False)
    
    # Grid-like graph
    multi_edges = [
        (1, 2), (2, 3), (3, 4),
        (1, 5), (2, 6), (3, 7), (4, 8),
        (5, 6), (6, 7), (7, 8)
    ]
    
    for u, v in multi_edges:
        multi_bfs_graph.add_edge(u, v)
    
    print("Graph for multi-source BFS:")
    multi_bfs_graph.display()
    
    # Multiple fire sources
    sources = [1, 8]  # Fire starts at positions 1 and 8
    distances = multi_bfs_graph.multi_source_bfs(sources)
    print(f"Distances from fire sources {sources}:")
    for vertex, distance in sorted(distances.items()):
        print(f"  Vertex {vertex}: {distance} steps") 