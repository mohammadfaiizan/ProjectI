"""
Graph Representations - Comprehensive Implementation
This module covers different ways to represent graphs and various graph types.
"""

from collections import defaultdict
import numpy as np

class GraphRepresentations:
    
    # ==================== ADJACENCY LIST REPRESENTATION ====================
    
    class AdjacencyList:
        def __init__(self, directed=False, weighted=False):
            self.graph = defaultdict(list)
            self.directed = directed
            self.weighted = weighted
            self.vertices = set()
        
        def add_vertex(self, vertex):
            """Add a vertex to the graph"""
            self.vertices.add(vertex)
            if vertex not in self.graph:
                self.graph[vertex] = []
        
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
        
        def remove_edge(self, u, v):
            """Remove an edge from the graph"""
            if self.weighted:
                self.graph[u] = [(vertex, weight) for vertex, weight in self.graph[u] if vertex != v]
                if not self.directed:
                    self.graph[v] = [(vertex, weight) for vertex, weight in self.graph[v] if vertex != u]
            else:
                if v in self.graph[u]:
                    self.graph[u].remove(v)
                if not self.directed and u in self.graph[v]:
                    self.graph[v].remove(u)
        
        def get_vertices(self):
            """Get all vertices in the graph"""
            return list(self.vertices)
        
        def get_edges(self):
            """Get all edges in the graph"""
            edges = []
            for u in self.graph:
                if self.weighted:
                    for v, weight in self.graph[u]:
                        if self.directed or u <= v:  # Avoid duplicates in undirected graph
                            edges.append((u, v, weight))
                else:
                    for v in self.graph[u]:
                        if self.directed or u <= v:  # Avoid duplicates in undirected graph
                            edges.append((u, v))
            return edges
        
        def get_neighbors(self, vertex):
            """Get neighbors of a vertex"""
            return self.graph[vertex]
        
        def has_edge(self, u, v):
            """Check if edge exists between u and v"""
            if self.weighted:
                return any(vertex == v for vertex, _ in self.graph[u])
            else:
                return v in self.graph[u]
        
        def display(self):
            """Display the graph"""
            for vertex in self.graph:
                print(f"{vertex}: {self.graph[vertex]}")
    
    # ==================== ADJACENCY MATRIX REPRESENTATION ====================
    
    class AdjacencyMatrix:
        def __init__(self, num_vertices, directed=False, weighted=False):
            self.num_vertices = num_vertices
            self.directed = directed
            self.weighted = weighted
            if weighted:
                # Initialize with infinity for weighted graphs
                self.matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
                # Distance from vertex to itself is 0
                for i in range(num_vertices):
                    self.matrix[i][i] = 0
            else:
                self.matrix = [[0] * num_vertices for _ in range(num_vertices)]
        
        def add_edge(self, u, v, weight=1):
            """Add an edge to the graph"""
            if 0 <= u < self.num_vertices and 0 <= v < self.num_vertices:
                if self.weighted:
                    self.matrix[u][v] = weight
                    if not self.directed:
                        self.matrix[v][u] = weight
                else:
                    self.matrix[u][v] = 1
                    if not self.directed:
                        self.matrix[v][u] = 1
        
        def remove_edge(self, u, v):
            """Remove an edge from the graph"""
            if 0 <= u < self.num_vertices and 0 <= v < self.num_vertices:
                if self.weighted:
                    self.matrix[u][v] = float('inf')
                    if not self.directed:
                        self.matrix[v][u] = float('inf')
                else:
                    self.matrix[u][v] = 0
                    if not self.directed:
                        self.matrix[v][u] = 0
        
        def has_edge(self, u, v):
            """Check if edge exists between u and v"""
            if 0 <= u < self.num_vertices and 0 <= v < self.num_vertices:
                if self.weighted:
                    return self.matrix[u][v] != float('inf')
                else:
                    return self.matrix[u][v] == 1
            return False
        
        def get_weight(self, u, v):
            """Get weight of edge between u and v"""
            if self.has_edge(u, v):
                return self.matrix[u][v]
            return float('inf') if self.weighted else 0
        
        def get_neighbors(self, vertex):
            """Get neighbors of a vertex"""
            neighbors = []
            for i in range(self.num_vertices):
                if self.has_edge(vertex, i):
                    if self.weighted:
                        neighbors.append((i, self.matrix[vertex][i]))
                    else:
                        neighbors.append(i)
            return neighbors
        
        def display(self):
            """Display the adjacency matrix"""
            for row in self.matrix:
                print(row)
    
    # ==================== EDGE LIST REPRESENTATION ====================
    
    class EdgeList:
        def __init__(self, directed=False, weighted=False):
            self.edges = []
            self.directed = directed
            self.weighted = weighted
            self.vertices = set()
        
        def add_edge(self, u, v, weight=1):
            """Add an edge to the graph"""
            self.vertices.add(u)
            self.vertices.add(v)
            
            if self.weighted:
                self.edges.append((u, v, weight))
                if not self.directed:
                    self.edges.append((v, u, weight))
            else:
                self.edges.append((u, v))
                if not self.directed:
                    self.edges.append((v, u))
        
        def remove_edge(self, u, v):
            """Remove an edge from the graph"""
            if self.weighted:
                self.edges = [(x, y, w) for x, y, w in self.edges if not ((x == u and y == v) or (not self.directed and x == v and y == u))]
            else:
                self.edges = [(x, y) for x, y in self.edges if not ((x == u and y == v) or (not self.directed and x == v and y == u))]
        
        def has_edge(self, u, v):
            """Check if edge exists between u and v"""
            if self.weighted:
                return any((x == u and y == v) for x, y, _ in self.edges)
            else:
                return any((x == u and y == v) for x, y in self.edges)
        
        def get_edges(self):
            """Get all edges"""
            return self.edges
        
        def get_vertices(self):
            """Get all vertices"""
            return list(self.vertices)
        
        def display(self):
            """Display all edges"""
            print("Edges:", self.edges)

    # ==================== GRAPH TYPE CHECKERS ====================
    
    @staticmethod
    def is_connected_undirected(adj_list):
        """Check if undirected graph is connected using DFS"""
        if not adj_list.graph:
            return True
        
        visited = set()
        start_vertex = next(iter(adj_list.vertices))
        
        def dfs(v):
            visited.add(v)
            neighbors = adj_list.get_neighbors(v)
            for neighbor in neighbors:
                if adj_list.weighted:
                    neighbor = neighbor[0]  # Extract vertex from (vertex, weight) tuple
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(start_vertex)
        return len(visited) == len(adj_list.vertices)
    
    @staticmethod
    def is_strongly_connected_directed(adj_list):
        """Check if directed graph is strongly connected using Kosaraju's algorithm"""
        if not adj_list.graph:
            return True
        
        # Step 1: DFS on original graph
        visited = set()
        stack = []
        
        def dfs1(v):
            visited.add(v)
            neighbors = adj_list.get_neighbors(v)
            for neighbor in neighbors:
                if adj_list.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    dfs1(neighbor)
            stack.append(v)
        
        for vertex in adj_list.vertices:
            if vertex not in visited:
                dfs1(vertex)
        
        # Step 2: Create transpose graph
        transpose = GraphRepresentations.AdjacencyList(directed=True, weighted=adj_list.weighted)
        for u in adj_list.graph:
            neighbors = adj_list.get_neighbors(u)
            for neighbor in neighbors:
                if adj_list.weighted:
                    v, weight = neighbor
                    transpose.add_edge(v, u, weight)
                else:
                    transpose.add_edge(neighbor, u)
        
        # Step 3: DFS on transpose graph in reverse order
        visited.clear()
        scc_count = 0
        
        def dfs2(v):
            visited.add(v)
            neighbors = transpose.get_neighbors(v)
            for neighbor in neighbors:
                if transpose.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    dfs2(neighbor)
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                dfs2(vertex)
                scc_count += 1
        
        return scc_count == 1
    
    @staticmethod
    def has_cycle_undirected(adj_list):
        """Detect cycle in undirected graph using DFS"""
        visited = set()
        
        def dfs(v, parent):
            visited.add(v)
            neighbors = adj_list.get_neighbors(v)
            for neighbor in neighbors:
                if adj_list.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    if dfs(neighbor, v):
                        return True
                elif neighbor != parent:
                    return True
            return False
        
        for vertex in adj_list.vertices:
            if vertex not in visited:
                if dfs(vertex, -1):
                    return True
        return False
    
    @staticmethod
    def has_cycle_directed(adj_list):
        """Detect cycle in directed graph using DFS (3-color method)"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in adj_list.vertices}
        
        def dfs(v):
            color[v] = GRAY
            neighbors = adj_list.get_neighbors(v)
            for neighbor in neighbors:
                if adj_list.weighted:
                    neighbor = neighbor[0]
                if color[neighbor] == GRAY:  # Back edge found
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[v] = BLACK
            return False
        
        for vertex in adj_list.vertices:
            if color[vertex] == WHITE:
                if dfs(vertex):
                    return True
        return False
    
    @staticmethod
    def is_bipartite(adj_list):
        """Check if graph is bipartite using BFS coloring"""
        color = {}
        
        def bfs_color(start):
            from collections import deque
            queue = deque([start])
            color[start] = 0
            
            while queue:
                v = queue.popleft()
                neighbors = adj_list.get_neighbors(v)
                for neighbor in neighbors:
                    if adj_list.weighted:
                        neighbor = neighbor[0]
                    if neighbor not in color:
                        color[neighbor] = 1 - color[v]
                        queue.append(neighbor)
                    elif color[neighbor] == color[v]:
                        return False
            return True
        
        for vertex in adj_list.vertices:
            if vertex not in color:
                if not bfs_color(vertex):
                    return False
        return True


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Graph Representations Demo ===\n")
    
    # 1. Adjacency List - Undirected Unweighted
    print("1. Adjacency List - Undirected Unweighted:")
    adj_list = GraphRepresentations.AdjacencyList(directed=False, weighted=False)
    adj_list.add_edge(0, 1)
    adj_list.add_edge(1, 2)
    adj_list.add_edge(2, 3)
    adj_list.add_edge(3, 0)
    adj_list.display()
    print(f"Has edge (0,1): {adj_list.has_edge(0, 1)}")
    print(f"Neighbors of 1: {adj_list.get_neighbors(1)}")
    print()
    
    # 2. Adjacency List - Directed Weighted
    print("2. Adjacency List - Directed Weighted:")
    weighted_graph = GraphRepresentations.AdjacencyList(directed=True, weighted=True)
    weighted_graph.add_edge('A', 'B', 5)
    weighted_graph.add_edge('B', 'C', 3)
    weighted_graph.add_edge('C', 'A', 2)
    weighted_graph.display()
    print()
    
    # 3. Adjacency Matrix
    print("3. Adjacency Matrix - Undirected:")
    adj_matrix = GraphRepresentations.AdjacencyMatrix(4, directed=False)
    adj_matrix.add_edge(0, 1)
    adj_matrix.add_edge(1, 2)
    adj_matrix.add_edge(2, 3)
    adj_matrix.display()
    print()
    
    # 4. Edge List
    print("4. Edge List - Undirected:")
    edge_list = GraphRepresentations.EdgeList(directed=False)
    edge_list.add_edge(0, 1)
    edge_list.add_edge(1, 2)
    edge_list.add_edge(2, 0)
    edge_list.display()
    print()
    
    # 5. Graph Properties Testing
    print("5. Graph Properties:")
    print(f"Is connected: {GraphRepresentations.is_connected_undirected(adj_list)}")
    print(f"Has cycle (undirected): {GraphRepresentations.has_cycle_undirected(adj_list)}")
    print(f"Is bipartite: {GraphRepresentations.is_bipartite(adj_list)}")
    
    # Test directed graph cycle
    directed_graph = GraphRepresentations.AdjacencyList(directed=True)
    directed_graph.add_edge(0, 1)
    directed_graph.add_edge(1, 2)
    directed_graph.add_edge(2, 0)  # Creates cycle
    print(f"Has cycle (directed): {GraphRepresentations.has_cycle_directed(directed_graph)}") 