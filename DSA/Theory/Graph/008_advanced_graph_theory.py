"""
Advanced Graph Theory Algorithms
This module implements advanced graph algorithms including Tarjan's algorithms,
strongly connected components, bridges, articulation points, and biconnected components.
"""

from collections import defaultdict, deque
import sys

class AdvancedGraphTheory:
    
    def __init__(self, directed=True):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.directed = directed
        self.time = 0  # Global time for DFS
    
    def add_edge(self, u, v):
        """Add an edge to the graph"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append(v)
        if not self.directed:
            self.graph[v].append(u)
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    # ==================== TARJAN'S ALGORITHM FOR BRIDGES ====================
    
    def find_bridges(self):
        """
        Find all bridges in an undirected graph using Tarjan's algorithm
        A bridge is an edge whose removal increases the number of connected components
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            list: List of bridge edges
        """
        if self.directed:
            raise ValueError("Bridge finding is for undirected graphs only")
        
        visited = set()
        discovery = {}  # Discovery time
        low = {}       # Low value
        parent = {}    # Parent in DFS tree
        bridges = []
        
        def bridge_dfs(u):
            self.time += 1
            visited.add(u)
            discovery[u] = low[u] = self.time
            
            for v in self.get_neighbors(u):
                if v not in visited:
                    parent[v] = u
                    bridge_dfs(v)
                    
                    # Update low value
                    low[u] = min(low[u], low[v])
                    
                    # If low value of v is more than discovery value of u,
                    # then u-v is a bridge
                    if low[v] > discovery[u]:
                        bridges.append((u, v))
                
                elif v != parent.get(u):  # Back edge
                    low[u] = min(low[u], discovery[v])
        
        self.time = 0
        for vertex in self.vertices:
            if vertex not in visited:
                parent[vertex] = None
                bridge_dfs(vertex)
        
        return bridges
    
    # ==================== TARJAN'S ALGORITHM FOR ARTICULATION POINTS ====================
    
    def find_articulation_points(self):
        """
        Find all articulation points (cut vertices) in an undirected graph
        An articulation point is a vertex whose removal increases the number of connected components
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            list: List of articulation points
        """
        if self.directed:
            raise ValueError("Articulation point finding is for undirected graphs only")
        
        visited = set()
        discovery = {}
        low = {}
        parent = {}
        articulation_points = set()
        
        def ap_dfs(u):
            children = 0
            self.time += 1
            visited.add(u)
            discovery[u] = low[u] = self.time
            
            for v in self.get_neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    ap_dfs(v)
                    
                    # Update low value
                    low[u] = min(low[u], low[v])
                    
                    # u is an articulation point in following cases:
                    
                    # Case 1: u is root of DFS tree and has two or more children
                    if parent.get(u) is None and children > 1:
                        articulation_points.add(u)
                    
                    # Case 2: u is not root and low value of one of its children
                    # is more than or equal to discovery value of u
                    if parent.get(u) is not None and low[v] >= discovery[u]:
                        articulation_points.add(u)
                
                elif v != parent.get(u):  # Back edge
                    low[u] = min(low[u], discovery[v])
        
        self.time = 0
        for vertex in self.vertices:
            if vertex not in visited:
                parent[vertex] = None
                ap_dfs(vertex)
        
        return list(articulation_points)
    
    # ==================== STRONGLY CONNECTED COMPONENTS - KOSARAJU'S ALGORITHM ====================
    
    def strongly_connected_components_kosaraju(self):
        """
        Find strongly connected components using Kosaraju's algorithm
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            list: List of SCCs, each SCC is a list of vertices
        """
        if not self.directed:
            raise ValueError("SCCs are defined for directed graphs only")
        
        # Step 1: Fill vertices in stack according to their finishing times
        visited = set()
        stack = []
        
        def fill_order(v):
            visited.add(v)
            for neighbor in self.get_neighbors(v):
                if neighbor not in visited:
                    fill_order(neighbor)
            stack.append(v)
        
        for vertex in self.vertices:
            if vertex not in visited:
                fill_order(vertex)
        
        # Step 2: Create transpose graph
        transpose = AdvancedGraphTheory(directed=True)
        for u in self.graph:
            for v in self.graph[u]:
                transpose.add_edge(v, u)  # Reverse the edge
        
        # Step 3: Process all vertices in order defined by stack
        visited.clear()
        sccs = []
        
        def dfs_scc(v, current_scc):
            visited.add(v)
            current_scc.append(v)
            for neighbor in transpose.get_neighbors(v):
                if neighbor not in visited:
                    dfs_scc(neighbor, current_scc)
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                current_scc = []
                dfs_scc(vertex, current_scc)
                sccs.append(current_scc)
        
        return sccs
    
    # ==================== STRONGLY CONNECTED COMPONENTS - TARJAN'S ALGORITHM ====================
    
    def strongly_connected_components_tarjan(self):
        """
        Find strongly connected components using Tarjan's algorithm
        
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Returns:
            list: List of SCCs, each SCC is a list of vertices
        """
        if not self.directed:
            raise ValueError("SCCs are defined for directed graphs only")
        
        index_counter = [0]  # Use list to make it mutable in nested function
        stack = []
        lowlink = {}
        index = {}
        on_stack = set()
        sccs = []
        
        def strongconnect(v):
            # Set the depth index for v to the smallest unused index
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)
            
            # Consider successors of v
            for w in self.get_neighbors(v):
                if w not in index:
                    # Successor w has not yet been visited; recurse on it
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    # Successor w is in stack and hence in the current SCC
                    lowlink[v] = min(lowlink[v], index[w])
            
            # If v is a root node, pop the stack and print an SCC
            if lowlink[v] == index[v]:
                current_scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    current_scc.append(w)
                    if w == v:
                        break
                sccs.append(current_scc)
        
        for vertex in self.vertices:
            if vertex not in index:
                strongconnect(vertex)
        
        return sccs
    
    # ==================== BICONNECTED COMPONENTS ====================
    
    def find_biconnected_components(self):
        """
        Find all biconnected components in an undirected graph
        A biconnected component is a maximal biconnected subgraph
        
        Time Complexity: O(V + E)
        Space Complexity: O(V + E)
        
        Returns:
            list: List of biconnected components, each as a list of edges
        """
        if self.directed:
            raise ValueError("Biconnected components are for undirected graphs only")
        
        visited = set()
        discovery = {}
        low = {}
        parent = {}
        stack = []  # Stack to store edges
        biconnected_components = []
        
        def bc_dfs(u):
            self.time += 1
            visited.add(u)
            discovery[u] = low[u] = self.time
            children = 0
            
            for v in self.get_neighbors(u):
                if v not in visited:
                    children += 1
                    parent[v] = u
                    stack.append((u, v))  # Push edge to stack
                    
                    bc_dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # If u is an articulation point, pop edges until (u, v)
                    if ((parent.get(u) is None and children > 1) or
                        (parent.get(u) is not None and low[v] >= discovery[u])):
                        
                        current_component = []
                        while True:
                            edge = stack.pop()
                            current_component.append(edge)
                            if edge == (u, v):
                                break
                        biconnected_components.append(current_component)
                
                elif v != parent.get(u) and discovery[v] < discovery[u]:
                    # Back edge
                    stack.append((u, v))
                    low[u] = min(low[u], discovery[v])
        
        self.time = 0
        for vertex in self.vertices:
            if vertex not in visited:
                parent[vertex] = None
                bc_dfs(vertex)
                
                # Pop remaining edges from stack (they form one component)
                if stack:
                    biconnected_components.append(list(stack))
                    stack.clear()
        
        return biconnected_components
    
    # ==================== ADVANCED TOPOLOGICAL SORTING ====================
    
    def topological_sort_with_cycle_detection(self):
        """
        Topological sort with detailed cycle detection and reporting
        
        Returns:
            tuple: (topological_order, has_cycle, cycle_info)
        """
        if not self.directed:
            raise ValueError("Topological sort is for directed graphs only")
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {vertex: WHITE for vertex in self.vertices}
        parent = {}
        topological_order = []
        cycle_info = None
        
        def dfs_topo(vertex):
            color[vertex] = GRAY
            
            for neighbor in self.get_neighbors(vertex):
                if color[neighbor] == GRAY:
                    # Back edge found - cycle detected
                    cycle = [neighbor]
                    current = vertex
                    while current != neighbor and current in parent:
                        cycle.append(current)
                        current = parent[current]
                    return cycle[::-1]
                
                if color[neighbor] == WHITE:
                    parent[neighbor] = vertex
                    result = dfs_topo(neighbor)
                    if result:  # Cycle found
                        return result
            
            color[vertex] = BLACK
            topological_order.append(vertex)
            return None
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                cycle = dfs_topo(vertex)
                if cycle:
                    cycle_info = {
                        'cycle': cycle,
                        'cycle_edges': [(cycle[i], cycle[(i + 1) % len(cycle)]) 
                                      for i in range(len(cycle))]
                    }
                    return [], True, cycle_info
        
        return topological_order[::-1], False, None
    
    def all_topological_sorts_with_analysis(self):
        """
        Find all possible topological sorts with analysis
        
        Returns:
            dict: Analysis including all sorts, count, and properties
        """
        if not self.directed:
            raise ValueError("Topological sort is for directed graphs only")
        
        # First check if DAG
        _, has_cycle, cycle_info = self.topological_sort_with_cycle_detection()
        if has_cycle:
            return {
                'all_sorts': [],
                'count': 0,
                'has_cycle': True,
                'cycle_info': cycle_info
            }
        
        # Calculate in-degrees
        in_degree = {vertex: 0 for vertex in self.vertices}
        for vertex in self.graph:
            for neighbor in self.graph[vertex]:
                in_degree[neighbor] += 1
        
        all_sorts = []
        
        def backtrack(current_sort, current_in_degree):
            # Find all vertices with in-degree 0
            available = [v for v in self.vertices 
                        if v not in current_sort and current_in_degree[v] == 0]
            
            if not available:
                if len(current_sort) == len(self.vertices):
                    all_sorts.append(current_sort.copy())
                return
            
            for vertex in available:
                # Choose vertex
                current_sort.append(vertex)
                
                # Update in-degrees
                new_in_degree = current_in_degree.copy()
                for neighbor in self.get_neighbors(vertex):
                    new_in_degree[neighbor] -= 1
                
                # Recurse
                backtrack(current_sort, new_in_degree)
                
                # Backtrack
                current_sort.pop()
        
        backtrack([], in_degree)
        
        # Analyze the results
        unique_first_vertices = set(sort[0] for sort in all_sorts if sort)
        unique_last_vertices = set(sort[-1] for sort in all_sorts if sort)
        
        return {
            'all_sorts': all_sorts,
            'count': len(all_sorts),
            'has_cycle': False,
            'unique_first_vertices': list(unique_first_vertices),
            'unique_last_vertices': list(unique_last_vertices),
            'lexicographically_first': min(all_sorts) if all_sorts else [],
            'lexicographically_last': max(all_sorts) if all_sorts else []
        }
    
    # ==================== GRAPH CONDENSATION ====================
    
    def condensation_graph(self):
        """
        Create condensation graph from SCCs
        In the condensation graph, each SCC becomes a single vertex
        
        Returns:
            tuple: (condensation_graph, scc_mapping)
        """
        if not self.directed:
            raise ValueError("Condensation is for directed graphs only")
        
        # Find SCCs
        sccs = self.strongly_connected_components_tarjan()
        
        # Create mapping from vertex to SCC index
        vertex_to_scc = {}
        for i, scc in enumerate(sccs):
            for vertex in scc:
                vertex_to_scc[vertex] = i
        
        # Build condensation graph
        condensation = AdvancedGraphTheory(directed=True)
        for i in range(len(sccs)):
            condensation.vertices.add(i)
        
        edges_added = set()
        for u in self.graph:
            scc_u = vertex_to_scc[u]
            for v in self.graph[u]:
                scc_v = vertex_to_scc[v]
                if scc_u != scc_v and (scc_u, scc_v) not in edges_added:
                    condensation.add_edge(scc_u, scc_v)
                    edges_added.add((scc_u, scc_v))
        
        return condensation, {i: scc for i, scc in enumerate(sccs)}
    
    # ==================== UTILITY METHODS ====================
    
    def display(self):
        """Display the graph"""
        for vertex in sorted(self.graph.keys()):
            neighbors = list(self.graph[vertex])
            print(f"{vertex}: {neighbors}")
    
    def get_graph_info(self):
        """Get comprehensive information about the graph"""
        num_vertices = len(self.vertices)
        num_edges = sum(len(neighbors) for neighbors in self.graph.values())
        if not self.directed:
            num_edges //= 2
        
        info = {
            "vertices": num_vertices,
            "edges": num_edges,
            "directed": self.directed,
            "vertices_list": sorted(self.vertices)
        }
        
        if self.directed:
            sccs = self.strongly_connected_components_tarjan()
            info["strongly_connected_components"] = len(sccs)
            info["is_strongly_connected"] = len(sccs) == 1
        else:
            bridges = self.find_bridges()
            aps = self.find_articulation_points()
            bcs = self.find_biconnected_components()
            info["bridges"] = len(bridges)
            info["articulation_points"] = len(aps)
            info["biconnected_components"] = len(bcs)
        
        return info
    
    def reset_time(self):
        """Reset the global time counter"""
        self.time = 0


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Advanced Graph Theory Algorithms Demo ===\n")
    
    # Example 1: Bridges and Articulation Points in Undirected Graph
    print("1. Bridges and Articulation Points:")
    undirected_graph = AdvancedGraphTheory(directed=False)
    
    # Create a graph with bridges and articulation points
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (4, 5), (5, 6), (4, 6)]
    for u, v in edges:
        undirected_graph.add_edge(u, v)
    
    print("Undirected Graph:")
    undirected_graph.display()
    
    bridges = undirected_graph.find_bridges()
    articulation_points = undirected_graph.find_articulation_points()
    biconnected_components = undirected_graph.find_biconnected_components()
    
    print(f"Bridges: {bridges}")
    print(f"Articulation Points: {articulation_points}")
    print(f"Biconnected Components: {biconnected_components}")
    print()
    
    # Example 2: Strongly Connected Components in Directed Graph
    print("2. Strongly Connected Components:")
    directed_graph = AdvancedGraphTheory(directed=True)
    
    # Create a directed graph with SCCs
    directed_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    for u, v in directed_edges:
        directed_graph.add_edge(u, v)
    
    print("Directed Graph:")
    directed_graph.display()
    
    # Kosaraju's Algorithm
    sccs_kosaraju = directed_graph.strongly_connected_components_kosaraju()
    print(f"SCCs (Kosaraju): {sccs_kosaraju}")
    
    # Tarjan's Algorithm
    sccs_tarjan = directed_graph.strongly_connected_components_tarjan()
    print(f"SCCs (Tarjan): {sccs_tarjan}")
    
    # Condensation Graph
    condensation, scc_mapping = directed_graph.condensation_graph()
    print(f"Condensation Graph:")
    condensation.display()
    print(f"SCC Mapping: {scc_mapping}")
    print()
    
    # Example 3: Advanced Topological Sorting
    print("3. Advanced Topological Sorting:")
    dag = AdvancedGraphTheory(directed=True)
    
    # Create a DAG
    dag_edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    for u, v in dag_edges:
        dag.add_edge(u, v)
    
    print("DAG:")
    dag.display()
    
    # Topological sort with cycle detection
    topo_order, has_cycle, cycle_info = dag.topological_sort_with_cycle_detection()
    print(f"Topological Order: {topo_order}")
    print(f"Has Cycle: {has_cycle}")
    
    # All topological sorts analysis
    analysis = dag.all_topological_sorts_with_analysis()
    print(f"All Topological Sorts Count: {analysis['count']}")
    print(f"First few sorts: {analysis['all_sorts'][:3]}")
    print(f"Unique first vertices: {analysis['unique_first_vertices']}")
    print(f"Unique last vertices: {analysis['unique_last_vertices']}")
    print()
    
    # Example 4: Cycle Detection in Directed Graph
    print("4. Cycle Detection in Directed Graph:")
    cyclic_graph = AdvancedGraphTheory(directed=True)
    
    # Create a graph with cycle
    cyclic_edges = [(0, 1), (1, 2), (2, 3), (3, 1), (0, 4)]
    for u, v in cyclic_edges:
        cyclic_graph.add_edge(u, v)
    
    print("Cyclic Directed Graph:")
    cyclic_graph.display()
    
    topo_order, has_cycle, cycle_info = cyclic_graph.topological_sort_with_cycle_detection()
    print(f"Has Cycle: {has_cycle}")
    if has_cycle:
        print(f"Cycle: {cycle_info['cycle']}")
        print(f"Cycle Edges: {cycle_info['cycle_edges']}")
    print()
    
    # Example 5: Complex Undirected Graph Analysis
    print("5. Complex Undirected Graph Analysis:")
    complex_graph = AdvancedGraphTheory(directed=False)
    
    # Create a more complex undirected graph
    complex_edges = [
        (0, 1), (1, 2), (2, 0),  # Triangle
        (1, 3), (3, 4),          # Bridge to another component
        (4, 5), (5, 6), (6, 4),  # Another triangle
        (7, 8)                   # Isolated edge
    ]
    for u, v in complex_edges:
        complex_graph.add_edge(u, v)
    
    print("Complex Undirected Graph:")
    complex_graph.display()
    
    info = complex_graph.get_graph_info()
    print(f"Graph Info: {info}")
    
    bridges = complex_graph.find_bridges()
    aps = complex_graph.find_articulation_points()
    bcs = complex_graph.find_biconnected_components()
    
    print(f"Bridges: {bridges}")
    print(f"Articulation Points: {aps}")
    print(f"Number of Biconnected Components: {len(bcs)}")
    print()
    
    # Example 6: Strongly Connected Graph
    print("6. Strongly Connected Graph Analysis:")
    strongly_connected = AdvancedGraphTheory(directed=True)
    
    # Create a strongly connected graph
    sc_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3), (2, 0)]
    for u, v in sc_edges:
        strongly_connected.add_edge(u, v)
    
    print("Strongly Connected Graph:")
    strongly_connected.display()
    
    sc_info = strongly_connected.get_graph_info()
    print(f"Is Strongly Connected: {sc_info['is_strongly_connected']}")
    print(f"Number of SCCs: {sc_info['strongly_connected_components']}")
    
    print("\n=== Demo Complete ===") 