"""
Cycle Detection Algorithms
This module implements various cycle detection algorithms for different types of graphs.
"""

from collections import defaultdict, deque

class CycleDetection:
    
    def __init__(self, directed=False, weighted=False):
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
    
    # ==================== UNION-FIND (DISJOINT SET UNION) ====================
    
    class UnionFind:
        """Union-Find data structure for cycle detection in undirected graphs"""
        
        def __init__(self, vertices):
            self.parent = {v: v for v in vertices}
            self.rank = {v: 0 for v in vertices}
        
        def find(self, x):
            """Find with path compression"""
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            """Union by rank"""
            root_x = self.find(x)
            root_y = self.find(y)
            
            if root_x == root_y:
                return False  # Cycle detected
            
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            
            return True
        
        def connected(self, x, y):
            """Check if two vertices are in the same component"""
            return self.find(x) == self.find(y)
    
    def has_cycle_union_find(self):
        """
        Detect cycle in undirected graph using Union-Find
        
        Time Complexity: O(E * α(V)) where α is inverse Ackermann function
        Space Complexity: O(V)
        
        Returns:
            bool: True if cycle exists, False otherwise
        """
        if self.directed:
            raise ValueError("Union-Find cycle detection is for undirected graphs only")
        
        uf = self.UnionFind(self.vertices)
        edges_processed = set()
        
        for u in self.graph:
            for neighbor_info in self.graph[u]:
                if self.weighted:
                    v, _ = neighbor_info
                else:
                    v = neighbor_info
                
                # Avoid processing the same edge twice in undirected graph
                edge = tuple(sorted([u, v]))
                if edge in edges_processed:
                    continue
                edges_processed.add(edge)
                
                # If vertices are already connected, adding this edge creates a cycle
                if not uf.union(u, v):
                    return True
        
        return False
    
    def find_cycle_union_find(self):
        """
        Find a cycle in undirected graph using Union-Find
        
        Returns:
            list: Edges that form the cycle, or None if no cycle
        """
        if self.directed:
            raise ValueError("Union-Find cycle detection is for undirected graphs only")
        
        uf = self.UnionFind(self.vertices)
        edges_processed = set()
        
        for u in self.graph:
            for neighbor_info in self.graph[u]:
                if self.weighted:
                    v, weight = neighbor_info
                else:
                    v, weight = neighbor_info, 1
                
                # Avoid processing the same edge twice
                edge = tuple(sorted([u, v]))
                if edge in edges_processed:
                    continue
                edges_processed.add(edge)
                
                # If vertices are already connected, this edge creates a cycle
                if not uf.union(u, v):
                    return (u, v, weight) if self.weighted else (u, v)
        
        return None
    
    # ==================== DFS-BASED CYCLE DETECTION ====================
    
    def has_cycle_dfs_undirected(self):
        """
        Detect cycle in undirected graph using DFS
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            bool: True if cycle exists, False otherwise
        """
        if self.directed:
            raise ValueError("This method is for undirected graphs only")
        
        visited = set()
        
        def dfs(vertex, parent):
            visited.add(vertex)
            
            neighbors = self.get_neighbors(vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                
                if neighbor not in visited:
                    if dfs(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True  # Back edge found (cycle)
            
            return False
        
        for vertex in self.vertices:
            if vertex not in visited:
                if dfs(vertex, None):
                    return True
        
        return False
    
    def has_cycle_dfs_directed(self):
        """
        Detect cycle in directed graph using DFS (3-color method)
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            bool: True if cycle exists, False otherwise
        """
        if not self.directed:
            raise ValueError("This method is for directed graphs only")
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        
        def dfs(vertex):
            color[vertex] = GRAY
            
            neighbors = self.get_neighbors(vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                
                if color[neighbor] == GRAY:  # Back edge found
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            
            color[vertex] = BLACK
            return False
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                if dfs(vertex):
                    return True
        
        return False
    
    def find_cycle_dfs_directed(self):
        """
        Find and return a cycle in directed graph using DFS
        
        Returns:
            list: Vertices forming a cycle, or None if no cycle
        """
        if not self.directed:
            raise ValueError("This method is for directed graphs only")
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        parent = {vertex: None for vertex in self.vertices}
        
        def dfs(vertex):
            color[vertex] = GRAY
            
            neighbors = self.get_neighbors(vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                
                if color[neighbor] == GRAY:  # Back edge found
                    # Reconstruct cycle
                    cycle = [neighbor]
                    current = vertex
                    while current != neighbor and current is not None:
                        cycle.append(current)
                        current = parent[current]
                    return cycle[::-1]
                
                if color[neighbor] == WHITE:
                    parent[neighbor] = vertex
                    result = dfs(neighbor)
                    if result:
                        return result
            
            color[vertex] = BLACK
            return None
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                cycle = dfs(vertex)
                if cycle:
                    return cycle
        
        return None
    
    def find_cycle_dfs_undirected(self):
        """
        Find a cycle in undirected graph using DFS
        
        Returns:
            list: Vertices forming a cycle, or None if no cycle
        """
        if self.directed:
            raise ValueError("This method is for undirected graphs only")
        
        visited = set()
        parent = {vertex: None for vertex in self.vertices}
        
        def dfs(vertex, par):
            visited.add(vertex)
            parent[vertex] = par
            
            neighbors = self.get_neighbors(vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                
                if neighbor not in visited:
                    result = dfs(neighbor, vertex)
                    if result:
                        return result
                elif neighbor != par:
                    # Cycle found, reconstruct it
                    cycle = [neighbor, vertex]
                    current = vertex
                    while parent[current] != neighbor and parent[current] is not None:
                        current = parent[current]
                        cycle.append(current)
                    return cycle
            
            return None
        
        for vertex in self.vertices:
            if vertex not in visited:
                cycle = dfs(vertex, None)
                if cycle:
                    return cycle
        
        return None
    
    # ==================== KAHN'S ALGORITHM FOR CYCLE DETECTION ====================
    
    def has_cycle_kahns(self):
        """
        Detect cycle in directed graph using Kahn's algorithm
        If topological sort is not possible, then cycle exists
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            bool: True if cycle exists, False otherwise
        """
        if not self.directed:
            raise ValueError("Kahn's algorithm is for directed graphs only")
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        
        for vertex in self.graph:
            neighbors = self.get_neighbors(vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                in_degree[neighbor] += 1
        
        # Initialize queue with vertices having in-degree 0
        queue = deque()
        for vertex in self.vertices:
            if in_degree[vertex] == 0:
                queue.append(vertex)
        
        processed_count = 0
        
        # Process vertices
        while queue:
            current = queue.popleft()
            processed_count += 1
            
            neighbors = self.get_neighbors(current)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If we couldn't process all vertices, there's a cycle
        return processed_count != len(self.vertices)
    
    def topological_sort_with_cycle_check(self):
        """
        Perform topological sort and detect cycles using Kahn's algorithm
        
        Returns:
            tuple: (topological_order, has_cycle)
        """
        if not self.directed:
            raise ValueError("Topological sort is for directed graphs only")
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        
        for vertex in self.graph:
            neighbors = self.get_neighbors(vertex)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                in_degree[neighbor] += 1
        
        # Initialize queue with vertices having in-degree 0
        queue = deque()
        for vertex in self.vertices:
            if in_degree[vertex] == 0:
                queue.append(vertex)
        
        topological_order = []
        
        # Process vertices
        while queue:
            current = queue.popleft()
            topological_order.append(current)
            
            neighbors = self.get_neighbors(current)
            for neighbor_info in neighbors:
                if self.weighted:
                    neighbor, _ = neighbor_info
                else:
                    neighbor = neighbor_info
                
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        has_cycle = len(topological_order) != len(self.vertices)
        
        return topological_order, has_cycle
    
    # ==================== UTILITY METHODS ====================
    
    def display(self):
        """Display the graph"""
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")
    
    def get_edges(self):
        """Get all edges in the graph"""
        edges = []
        processed = set()
        
        for u in self.graph:
            neighbors = self.get_neighbors(u)
            for neighbor_info in neighbors:
                if self.weighted:
                    v, weight = neighbor_info
                    edge = (u, v, weight)
                else:
                    v = neighbor_info
                    edge = (u, v)
                
                if not self.directed:
                    edge_key = tuple(sorted([u, v]))
                    if edge_key in processed:
                        continue
                    processed.add(edge_key)
                
                edges.append(edge)
        
        return edges
    
    def get_graph_info(self):
        """Get basic information about the graph"""
        num_vertices = len(self.vertices)
        num_edges = len(self.get_edges())
        
        return {
            "vertices": num_vertices,
            "edges": num_edges,
            "directed": self.directed,
            "weighted": self.weighted
        }


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Cycle Detection Algorithms Demo ===\n")
    
    # Example 1: Undirected Graph - Union-Find
    print("1. Undirected Graph Cycle Detection using Union-Find:")
    undirected_graph = CycleDetection(directed=False)
    
    # Add edges that form a cycle
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4)]
    for u, v in edges:
        undirected_graph.add_edge(u, v)
    
    print("Graph:")
    undirected_graph.display()
    print(f"Has cycle (Union-Find): {undirected_graph.has_cycle_union_find()}")
    print(f"Has cycle (DFS): {undirected_graph.has_cycle_dfs_undirected()}")
    print(f"Cycle found (Union-Find): {undirected_graph.find_cycle_union_find()}")
    print(f"Cycle found (DFS): {undirected_graph.find_cycle_dfs_undirected()}")
    print()
    
    # Example 2: Undirected Graph without cycle
    print("2. Undirected Graph without Cycle:")
    tree_graph = CycleDetection(directed=False)
    tree_edges = [(0, 1), (1, 2), (1, 3), (3, 4)]
    for u, v in tree_edges:
        tree_graph.add_edge(u, v)
    
    print("Tree Graph:")
    tree_graph.display()
    print(f"Has cycle (Union-Find): {tree_graph.has_cycle_union_find()}")
    print(f"Has cycle (DFS): {tree_graph.has_cycle_dfs_undirected()}")
    print()
    
    # Example 3: Directed Graph with cycle - DFS
    print("3. Directed Graph Cycle Detection using DFS:")
    directed_graph = CycleDetection(directed=True)
    
    # Add edges that form a cycle
    directed_edges = [(0, 1), (1, 2), (2, 3), (3, 1), (0, 4)]
    for u, v in directed_edges:
        directed_graph.add_edge(u, v)
    
    print("Directed Graph:")
    directed_graph.display()
    print(f"Has cycle (DFS): {directed_graph.has_cycle_dfs_directed()}")
    print(f"Cycle found (DFS): {directed_graph.find_cycle_dfs_directed()}")
    print()
    
    # Example 4: Directed Graph - Kahn's Algorithm
    print("4. Directed Graph Cycle Detection using Kahn's Algorithm:")
    print(f"Has cycle (Kahn's): {directed_graph.has_cycle_kahns()}")
    
    topo_order, has_cycle = directed_graph.topological_sort_with_cycle_check()
    print(f"Topological Order: {topo_order}")
    print(f"Has cycle: {has_cycle}")
    print()
    
    # Example 5: Directed Acyclic Graph (DAG)
    print("5. Directed Acyclic Graph (DAG):")
    dag = CycleDetection(directed=True)
    dag_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    for u, v in dag_edges:
        dag.add_edge(u, v)
    
    print("DAG:")
    dag.display()
    print(f"Has cycle (DFS): {dag.has_cycle_dfs_directed()}")
    print(f"Has cycle (Kahn's): {dag.has_cycle_kahns()}")
    
    topo_order, has_cycle = dag.topological_sort_with_cycle_check()
    print(f"Topological Order: {topo_order}")
    print(f"Has cycle: {has_cycle}")
    print()
    
    # Example 6: Weighted Graph
    print("6. Weighted Graph Cycle Detection:")
    weighted_graph = CycleDetection(directed=False, weighted=True)
    weighted_edges = [(0, 1, 5), (1, 2, 3), (2, 0, 2), (0, 3, 7)]
    for u, v, w in weighted_edges:
        weighted_graph.add_edge(u, v, w)
    
    print("Weighted Graph:")
    weighted_graph.display()
    print(f"Has cycle (Union-Find): {weighted_graph.has_cycle_union_find()}")
    print(f"Cycle found: {weighted_graph.find_cycle_union_find()}")
    print(f"Graph Info: {weighted_graph.get_graph_info()}") 