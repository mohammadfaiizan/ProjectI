"""
Graph Traversal Algorithms - DFS and BFS
This module implements various graph traversal techniques and their applications.
"""

from collections import defaultdict, deque

class GraphTraversal:
    
    def __init__(self, directed=False, weighted=False):
        self.graph = defaultdict(list)
        self.directed = directed
        self.weighted = weighted
        self.vertices = set()
    
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
    
    # ==================== DEPTH FIRST SEARCH (DFS) ====================
    
    def dfs_recursive(self, start_vertex, visited=None):
        """
        Recursive DFS traversal
        Returns list of vertices in DFS order
        """
        if visited is None:
            visited = set()
        
        result = []
        
        def dfs_helper(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]  # Extract vertex from (vertex, weight) tuple
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start_vertex)
        return result
    
    def dfs_iterative(self, start_vertex):
        """
        Iterative DFS traversal using stack
        Returns list of vertices in DFS order
        """
        visited = set()
        stack = [start_vertex]
        result = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add neighbors to stack (in reverse order to maintain left-to-right traversal)
                neighbors = self.get_neighbors(vertex)
                neighbor_list = []
                for neighbor in neighbors:
                    if self.weighted:
                        neighbor_list.append(neighbor[0])
                    else:
                        neighbor_list.append(neighbor)
                
                # Add in reverse order so we visit left neighbors first
                for neighbor in reversed(neighbor_list):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    def dfs_with_path(self, start, end, visited=None, path=None):
        """
        DFS to find path between two vertices
        Returns the path if exists, None otherwise
        """
        if visited is None:
            visited = set()
        if path is None:
            path = []
        
        visited.add(start)
        path.append(start)
        
        if start == end:
            return path.copy()
        
        neighbors = self.get_neighbors(start)
        for neighbor in neighbors:
            if self.weighted:
                neighbor = neighbor[0]
            if neighbor not in visited:
                result = self.dfs_with_path(neighbor, end, visited, path)
                if result:
                    return result
        
        path.pop()
        return None
    
    def dfs_all_paths(self, start, end, visited=None, path=None, all_paths=None):
        """
        Find all paths between two vertices using DFS
        Returns list of all possible paths
        """
        if visited is None:
            visited = set()
        if path is None:
            path = []
        if all_paths is None:
            all_paths = []
        
        visited.add(start)
        path.append(start)
        
        if start == end:
            all_paths.append(path.copy())
        else:
            neighbors = self.get_neighbors(start)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    self.dfs_all_paths(neighbor, end, visited, path, all_paths)
        
        # Backtrack
        visited.remove(start)
        path.pop()
        
        return all_paths
    
    # ==================== BREADTH FIRST SEARCH (BFS) ====================
    
    def bfs(self, start_vertex):
        """
        BFS traversal using queue
        Returns list of vertices in BFS order
        """
        visited = set()
        queue = deque([start_vertex])
        result = []
        
        visited.add(start_vertex)
        
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result
    
    def bfs_shortest_path(self, start, end):
        """
        Find shortest path (in terms of edges) between two vertices using BFS
        Returns path and distance
        """
        if start == end:
            return [start], 0
        
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        
        while queue:
            vertex, path = queue.popleft()
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if neighbor == end:
                    return path + [neighbor], len(path)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None, -1  # No path found
    
    def bfs_level_order(self, start_vertex):
        """
        BFS with level information
        Returns list of levels, where each level contains vertices at that distance
        """
        visited = set()
        queue = deque([(start_vertex, 0)])
        levels = {}
        
        visited.add(start_vertex)
        
        while queue:
            vertex, level = queue.popleft()
            
            if level not in levels:
                levels[level] = []
            levels[level].append(vertex)
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, level + 1))
        
        return [levels[i] for i in sorted(levels.keys())]
    
    # ==================== CONNECTED COMPONENTS ====================
    
    def find_connected_components(self):
        """
        Find all connected components in the graph
        Returns list of components, where each component is a list of vertices
        """
        visited = set()
        components = []
        
        for vertex in self.vertices:
            if vertex not in visited:
                component = []
                self._dfs_component(vertex, visited, component)
                components.append(component)
        
        return components
    
    def _dfs_component(self, vertex, visited, component):
        """Helper method for finding connected components"""
        visited.add(vertex)
        component.append(vertex)
        
        neighbors = self.get_neighbors(vertex)
        for neighbor in neighbors:
            if self.weighted:
                neighbor = neighbor[0]
            if neighbor not in visited:
                self._dfs_component(neighbor, visited, component)
    
    def count_connected_components(self):
        """Count number of connected components"""
        return len(self.find_connected_components())
    
    def is_connected(self):
        """Check if the graph is connected"""
        return self.count_connected_components() <= 1
    
    # ==================== CYCLE DETECTION ====================
    
    def has_cycle_undirected(self):
        """
        Detect cycle in undirected graph using DFS
        Returns True if cycle exists, False otherwise
        """
        if self.directed:
            raise ValueError("This method is for undirected graphs only")
        
        visited = set()
        
        def dfs_cycle(vertex, parent):
            visited.add(vertex)
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    if dfs_cycle(neighbor, vertex):
                        return True
                elif neighbor != parent:
                    return True
            
            return False
        
        for vertex in self.vertices:
            if vertex not in visited:
                if dfs_cycle(vertex, -1):
                    return True
        
        return False
    
    def has_cycle_directed(self):
        """
        Detect cycle in directed graph using DFS (3-color method)
        Returns True if cycle exists, False otherwise
        """
        if not self.directed:
            raise ValueError("This method is for directed graphs only")
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        
        def dfs_cycle(vertex):
            color[vertex] = GRAY
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if color[neighbor] == GRAY:  # Back edge found
                    return True
                if color[neighbor] == WHITE and dfs_cycle(neighbor):
                    return True
            
            color[vertex] = BLACK
            return False
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                if dfs_cycle(vertex):
                    return True
        
        return False
    
    # ==================== BIPARTITE CHECK ====================
    
    def is_bipartite(self):
        """
        Check if graph is bipartite using BFS coloring
        Returns (is_bipartite, coloring) where coloring is a dict of vertex->color
        """
        color = {}
        
        def bfs_bipartite(start):
            queue = deque([start])
            color[start] = 0
            
            while queue:
                vertex = queue.popleft()
                
                neighbors = self.get_neighbors(vertex)
                for neighbor in neighbors:
                    if self.weighted:
                        neighbor = neighbor[0]
                    if neighbor not in color:
                        color[neighbor] = 1 - color[vertex]
                        queue.append(neighbor)
                    elif color[neighbor] == color[vertex]:
                        return False
            
            return True
        
        for vertex in self.vertices:
            if vertex not in color:
                if not bfs_bipartite(vertex):
                    return False, {}
        
        return True, color
    
    # ==================== TOPOLOGICAL SORT (Preview) ====================
    
    def topological_sort_dfs(self):
        """
        Topological sort using DFS (for directed acyclic graphs)
        Returns topologically sorted list of vertices
        """
        if not self.directed:
            raise ValueError("Topological sort is only for directed graphs")
        
        visited = set()
        stack = []
        
        def dfs_topo(vertex):
            visited.add(vertex)
            
            neighbors = self.get_neighbors(vertex)
            for neighbor in neighbors:
                if self.weighted:
                    neighbor = neighbor[0]
                if neighbor not in visited:
                    dfs_topo(neighbor)
            
            stack.append(vertex)
        
        for vertex in self.vertices:
            if vertex not in visited:
                dfs_topo(vertex)
        
        return stack[::-1]  # Reverse the stack
    
    # ==================== UTILITY METHODS ====================
    
    def display(self):
        """Display the graph"""
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")
    
    def get_graph_info(self):
        """Get basic information about the graph"""
        num_vertices = len(self.vertices)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values())
        if not self.directed:
            num_edges //= 2  # Each edge counted twice in undirected graph
        
        return {
            "vertices": num_vertices,
            "edges": num_edges,
            "directed": self.directed,
            "weighted": self.weighted,
            "connected": self.is_connected(),
            "components": self.count_connected_components()
        }


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Graph Traversal Algorithms Demo ===\n")
    
    # Create an undirected graph
    print("1. Undirected Graph Traversal:")
    graph = GraphTraversal(directed=False)
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
    for u, v in edges:
        graph.add_edge(u, v)
    
    print("Graph:")
    graph.display()
    print(f"Graph Info: {graph.get_graph_info()}")
    print()
    
    # DFS traversals
    print("DFS Recursive from 0:", graph.dfs_recursive(0))
    print("DFS Iterative from 0:", graph.dfs_iterative(0))
    print("Path from 0 to 5:", graph.dfs_with_path(0, 5))
    print("All paths from 0 to 5:", graph.dfs_all_paths(0, 5))
    print()
    
    # BFS traversals
    print("BFS from 0:", graph.bfs(0))
    path, distance = graph.bfs_shortest_path(0, 5)
    print(f"Shortest path from 0 to 5: {path}, Distance: {distance}")
    print("BFS Level Order from 0:", graph.bfs_level_order(0))
    print()
    
    # Graph properties
    print("Connected Components:", graph.find_connected_components())
    print("Has Cycle:", graph.has_cycle_undirected())
    is_bip, coloring = graph.is_bipartite()
    print(f"Is Bipartite: {is_bip}, Coloring: {coloring}")
    print()
    
    # Directed graph example
    print("2. Directed Graph Example:")
    directed_graph = GraphTraversal(directed=True)
    directed_edges = [(0, 1), (1, 2), (2, 3), (3, 1), (0, 4)]
    for u, v in directed_edges:
        directed_graph.add_edge(u, v)
    
    print("Directed Graph:")
    directed_graph.display()
    print("DFS from 0:", directed_graph.dfs_recursive(0))
    print("BFS from 0:", directed_graph.bfs(0))
    print("Has Cycle:", directed_graph.has_cycle_directed())
    print("Topological Sort:", directed_graph.topological_sort_dfs())
    print()
    
    # Disconnected graph example
    print("3. Disconnected Graph Example:")
    disconnected_graph = GraphTraversal(directed=False)
    disconnected_edges = [(0, 1), (2, 3), (4, 5)]
    for u, v in disconnected_edges:
        disconnected_graph.add_edge(u, v)
    
    print("Disconnected Graph:")
    disconnected_graph.display()
    print("Connected Components:", disconnected_graph.find_connected_components())
    print("Number of Components:", disconnected_graph.count_connected_components())
    print("Is Connected:", disconnected_graph.is_connected()) 