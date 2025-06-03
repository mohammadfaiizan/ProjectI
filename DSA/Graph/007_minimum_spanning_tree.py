"""
Minimum Spanning Tree (MST) Algorithms
This module implements Kruskal's and Prim's algorithms for finding MST.
"""

from collections import defaultdict
import heapq

class MinimumSpanningTree:
    
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.directed = directed
        if directed:
            raise ValueError("MST algorithms are for undirected graphs only")
    
    def add_edge(self, u, v, weight):
        """Add a weighted edge to the graph"""
        self.vertices.add(u)
        self.vertices.add(v)
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))  # Undirected graph
    
    def get_neighbors(self, vertex):
        """Get neighbors of a vertex"""
        return self.graph[vertex]
    
    def get_all_edges(self):
        """Get all edges in the graph"""
        edges = []
        processed = set()
        
        for u in self.graph:
            for v, weight in self.graph[u]:
                edge_key = tuple(sorted([u, v]))
                if edge_key not in processed:
                    edges.append((u, v, weight))
                    processed.add(edge_key)
        
        return edges
    
    # ==================== UNION-FIND FOR KRUSKAL'S ALGORITHM ====================
    
    class UnionFind:
        """Union-Find data structure for Kruskal's algorithm"""
        
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
            """Check if two vertices are connected"""
            return self.find(x) == self.find(y)
    
    # ==================== KRUSKAL'S ALGORITHM ====================
    
    def kruskals_mst(self):
        """
        Kruskal's Algorithm for Minimum Spanning Tree
        Uses Union-Find to detect cycles
        
        Time Complexity: O(E log E)
        Space Complexity: O(V)
        
        Returns:
            tuple: (mst_edges, total_weight, mst_graph)
        """
        # Get all edges and sort by weight
        edges = self.get_all_edges()
        edges.sort(key=lambda x: x[2])  # Sort by weight
        
        # Initialize Union-Find
        uf = self.UnionFind(self.vertices)
        
        mst_edges = []
        total_weight = 0
        mst_graph = defaultdict(list)
        
        # Process edges in sorted order
        for u, v, weight in edges:
            # If adding this edge doesn't create a cycle
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight
                
                # Add to MST graph representation
                mst_graph[u].append((v, weight))
                mst_graph[v].append((u, weight))
                
                # Stop when we have V-1 edges
                if len(mst_edges) == len(self.vertices) - 1:
                    break
        
        return mst_edges, total_weight, dict(mst_graph)
    
    def kruskals_mst_detailed(self):
        """
        Kruskal's Algorithm with detailed step-by-step information
        
        Returns:
            tuple: (mst_edges, total_weight, steps)
        """
        edges = self.get_all_edges()
        edges.sort(key=lambda x: x[2])
        
        uf = self.UnionFind(self.vertices)
        mst_edges = []
        total_weight = 0
        steps = []
        
        for i, (u, v, weight) in enumerate(edges):
            step_info = {
                'step': i + 1,
                'edge': (u, v, weight),
                'before_union': uf.connected(u, v),
                'action': None,
                'mst_edges_so_far': len(mst_edges),
                'total_weight_so_far': total_weight
            }
            
            if uf.union(u, v):
                mst_edges.append((u, v, weight))
                total_weight += weight
                step_info['action'] = 'added'
                step_info['mst_edges_so_far'] = len(mst_edges)
                step_info['total_weight_so_far'] = total_weight
            else:
                step_info['action'] = 'rejected (cycle)'
            
            steps.append(step_info)
            
            # Stop when MST is complete
            if len(mst_edges) == len(self.vertices) - 1:
                break
        
        return mst_edges, total_weight, steps
    
    # ==================== PRIM'S ALGORITHM ====================
    
    def prims_mst(self, start_vertex=None):
        """
        Prim's Algorithm for Minimum Spanning Tree
        Uses min-heap for efficient edge selection
        
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V + E)
        
        Args:
            start_vertex: Starting vertex (optional)
        
        Returns:
            tuple: (mst_edges, total_weight, mst_graph)
        """
        if not self.vertices:
            return [], 0, {}
        
        # Choose starting vertex
        if start_vertex is None:
            start_vertex = next(iter(self.vertices))
        elif start_vertex not in self.vertices:
            raise ValueError(f"Start vertex {start_vertex} not in graph")
        
        mst_edges = []
        total_weight = 0
        mst_graph = defaultdict(list)
        visited = set()
        
        # Priority queue: (weight, u, v)
        # Start with all edges from start_vertex
        heap = []
        visited.add(start_vertex)
        
        for neighbor, weight in self.get_neighbors(start_vertex):
            heapq.heappush(heap, (weight, start_vertex, neighbor))
        
        while heap and len(mst_edges) < len(self.vertices) - 1:
            weight, u, v = heapq.heappop(heap)
            
            # Skip if both vertices are already in MST
            if v in visited:
                continue
            
            # Add edge to MST
            mst_edges.append((u, v, weight))
            total_weight += weight
            visited.add(v)
            
            # Add to MST graph representation
            mst_graph[u].append((v, weight))
            mst_graph[v].append((u, weight))
            
            # Add all edges from the new vertex to unvisited vertices
            for neighbor, edge_weight in self.get_neighbors(v):
                if neighbor not in visited:
                    heapq.heappush(heap, (edge_weight, v, neighbor))
        
        return mst_edges, total_weight, dict(mst_graph)
    
    def prims_mst_detailed(self, start_vertex=None):
        """
        Prim's Algorithm with detailed step-by-step information
        
        Returns:
            tuple: (mst_edges, total_weight, steps)
        """
        if not self.vertices:
            return [], 0, []
        
        if start_vertex is None:
            start_vertex = next(iter(self.vertices))
        
        mst_edges = []
        total_weight = 0
        visited = set()
        heap = []
        steps = []
        
        # Start with the starting vertex
        visited.add(start_vertex)
        
        for neighbor, weight in self.get_neighbors(start_vertex):
            heapq.heappush(heap, (weight, start_vertex, neighbor))
        
        step_count = 0
        
        while heap and len(mst_edges) < len(self.vertices) - 1:
            step_count += 1
            weight, u, v = heapq.heappop(heap)
            
            step_info = {
                'step': step_count,
                'edge_considered': (u, v, weight),
                'visited_before': list(visited),
                'action': None,
                'mst_edges_so_far': len(mst_edges),
                'total_weight_so_far': total_weight
            }
            
            if v in visited:
                step_info['action'] = 'rejected (already visited)'
            else:
                mst_edges.append((u, v, weight))
                total_weight += weight
                visited.add(v)
                step_info['action'] = 'added'
                step_info['mst_edges_so_far'] = len(mst_edges)
                step_info['total_weight_so_far'] = total_weight
                
                # Add new edges to heap
                for neighbor, edge_weight in self.get_neighbors(v):
                    if neighbor not in visited:
                        heapq.heappush(heap, (edge_weight, v, neighbor))
            
            step_info['visited_after'] = list(visited)
            steps.append(step_info)
        
        return mst_edges, total_weight, steps
    
    def prims_mst_matrix(self, start_vertex=None):
        """
        Prim's Algorithm using adjacency matrix representation
        Alternative implementation for dense graphs
        
        Time Complexity: O(V²)
        Space Complexity: O(V²)
        
        Returns:
            tuple: (mst_edges, total_weight)
        """
        if not self.vertices:
            return [], 0
        
        vertices = list(self.vertices)
        n = len(vertices)
        vertex_to_index = {v: i for i, v in enumerate(vertices)}
        
        # Create adjacency matrix
        INF = float('inf')
        adj_matrix = [[INF] * n for _ in range(n)]
        
        for u in self.graph:
            u_idx = vertex_to_index[u]
            for v, weight in self.graph[u]:
                v_idx = vertex_to_index[v]
                adj_matrix[u_idx][v_idx] = weight
        
        # Prim's algorithm
        if start_vertex is None:
            start_idx = 0
        else:
            start_idx = vertex_to_index[start_vertex]
        
        key = [INF] * n  # Key values to pick minimum weight edge
        parent = [-1] * n  # Array to store MST
        in_mst = [False] * n  # Track vertices included in MST
        
        key[start_idx] = 0  # Start with the first vertex
        
        mst_edges = []
        total_weight = 0
        
        for _ in range(n):
            # Find minimum key vertex not yet in MST
            min_key = INF
            min_index = -1
            
            for v in range(n):
                if not in_mst[v] and key[v] < min_key:
                    min_key = key[v]
                    min_index = v
            
            if min_index == -1:
                break
            
            # Add vertex to MST
            in_mst[min_index] = True
            
            # Add edge to MST (except for the first vertex)
            if parent[min_index] != -1:
                u_vertex = vertices[parent[min_index]]
                v_vertex = vertices[min_index]
                weight = adj_matrix[parent[min_index]][min_index]
                mst_edges.append((u_vertex, v_vertex, weight))
                total_weight += weight
            
            # Update key values of adjacent vertices
            for v in range(n):
                if (not in_mst[v] and 
                    adj_matrix[min_index][v] != INF and 
                    adj_matrix[min_index][v] < key[v]):
                    key[v] = adj_matrix[min_index][v]
                    parent[v] = min_index
        
        return mst_edges, total_weight
    
    # ==================== MST VERIFICATION AND ANALYSIS ====================
    
    def verify_mst(self, mst_edges):
        """
        Verify if given edges form a valid MST
        
        Args:
            mst_edges: List of edges in MST
        
        Returns:
            tuple: (is_valid, issues)
        """
        issues = []
        
        # Check if it's a spanning tree (connects all vertices)
        if len(mst_edges) != len(self.vertices) - 1:
            issues.append(f"Wrong number of edges: {len(mst_edges)}, expected {len(self.vertices) - 1}")
        
        # Check if all edges are in the original graph
        original_edges = set()
        for u, v, w in self.get_all_edges():
            original_edges.add((min(u, v), max(u, v), w))
        
        for u, v, w in mst_edges:
            edge = (min(u, v), max(u, v), w)
            if edge not in original_edges:
                issues.append(f"Edge {(u, v, w)} not in original graph")
        
        # Check connectivity using Union-Find
        uf = self.UnionFind(self.vertices)
        for u, v, _ in mst_edges:
            uf.union(u, v)
        
        # Check if all vertices are connected
        root = uf.find(next(iter(self.vertices)))
        for vertex in self.vertices:
            if uf.find(vertex) != root:
                issues.append("MST doesn't connect all vertices")
                break
        
        return len(issues) == 0, issues
    
    def compare_algorithms(self, start_vertex=None):
        """
        Compare Kruskal's and Prim's algorithms
        
        Returns:
            dict: Comparison results
        """
        # Run Kruskal's
        kruskal_edges, kruskal_weight, _ = self.kruskals_mst()
        
        # Run Prim's
        prim_edges, prim_weight, _ = self.prims_mst(start_vertex)
        
        # Verify both results
        kruskal_valid, kruskal_issues = self.verify_mst(kruskal_edges)
        prim_valid, prim_issues = self.verify_mst(prim_edges)
        
        return {
            'kruskal': {
                'edges': kruskal_edges,
                'weight': kruskal_weight,
                'valid': kruskal_valid,
                'issues': kruskal_issues
            },
            'prim': {
                'edges': prim_edges,
                'weight': prim_weight,
                'valid': prim_valid,
                'issues': prim_issues
            },
            'weights_match': kruskal_weight == prim_weight
        }
    
    # ==================== UTILITY METHODS ====================
    
    def display(self):
        """Display the graph"""
        for vertex in sorted(self.graph.keys()):
            neighbors = [(v, w) for v, w in self.graph[vertex]]
            print(f"{vertex}: {neighbors}")
    
    def display_mst(self, mst_edges, total_weight):
        """Display MST edges and total weight"""
        print(f"MST Edges (Total Weight: {total_weight}):")
        for i, (u, v, weight) in enumerate(mst_edges, 1):
            print(f"  {i}. {u} -- {v} : {weight}")
    
    def get_graph_info(self):
        """Get basic information about the graph"""
        edges = self.get_all_edges()
        total_weight = sum(weight for _, _, weight in edges)
        
        return {
            "vertices": len(self.vertices),
            "edges": len(edges),
            "total_weight": total_weight,
            "vertices_list": sorted(self.vertices)
        }


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Minimum Spanning Tree Algorithms Demo ===\n")
    
    # Example 1: Basic MST Construction
    print("1. Basic MST Construction:")
    mst_graph = MinimumSpanningTree()
    
    # Add edges to create a connected graph
    edges = [
        ('A', 'B', 4), ('A', 'H', 8), ('B', 'C', 8), ('B', 'H', 11),
        ('C', 'D', 7), ('C', 'I', 2), ('C', 'F', 4), ('D', 'E', 9),
        ('D', 'F', 14), ('E', 'F', 10), ('F', 'G', 2), ('G', 'H', 1),
        ('G', 'I', 6), ('H', 'I', 7)
    ]
    
    for u, v, w in edges:
        mst_graph.add_edge(u, v, w)
    
    print("Original Graph:")
    mst_graph.display()
    print(f"Graph Info: {mst_graph.get_graph_info()}")
    print()
    
    # Run Kruskal's Algorithm
    print("2. Kruskal's Algorithm:")
    kruskal_edges, kruskal_weight, kruskal_mst = mst_graph.kruskals_mst()
    mst_graph.display_mst(kruskal_edges, kruskal_weight)
    print()
    
    # Run Prim's Algorithm
    print("3. Prim's Algorithm (starting from 'A'):")
    prim_edges, prim_weight, prim_mst = mst_graph.prims_mst('A')
    mst_graph.display_mst(prim_edges, prim_weight)
    print()
    
    # Detailed Kruskal's execution
    print("4. Detailed Kruskal's Algorithm Execution:")
    _, _, kruskal_steps = mst_graph.kruskals_mst_detailed()
    for step in kruskal_steps[:10]:  # Show first 10 steps
        edge = step['edge']
        action = step['action']
        print(f"Step {step['step']}: Edge {edge} -> {action}")
        if action == 'added':
            print(f"  MST edges so far: {step['mst_edges_so_far']}, Weight: {step['total_weight_so_far']}")
    print()
    
    # Compare algorithms
    print("5. Algorithm Comparison:")
    comparison = mst_graph.compare_algorithms('A')
    print(f"Kruskal's weight: {comparison['kruskal']['weight']}")
    print(f"Prim's weight: {comparison['prim']['weight']}")
    print(f"Weights match: {comparison['weights_match']}")
    print(f"Kruskal's valid: {comparison['kruskal']['valid']}")
    print(f"Prim's valid: {comparison['prim']['valid']}")
    print()
    
    # Example 2: Different graph
    print("6. Different Graph Example:")
    graph2 = MinimumSpanningTree()
    
    # Create a different graph
    edges2 = [
        (0, 1, 10), (0, 2, 6), (0, 3, 5),
        (1, 3, 15), (2, 3, 4)
    ]
    
    for u, v, w in edges2:
        graph2.add_edge(u, v, w)
    
    print("Graph 2:")
    graph2.display()
    
    # MST using matrix-based Prim's
    matrix_edges, matrix_weight = graph2.prims_mst_matrix(0)
    print(f"MST using matrix-based Prim's:")
    graph2.display_mst(matrix_edges, matrix_weight)
    print()
    
    # Example 3: Detailed Prim's execution
    print("7. Detailed Prim's Algorithm Execution:")
    _, _, prim_steps = graph2.prims_mst_detailed(0)
    for step in prim_steps:
        edge = step['edge_considered']
        action = step['action']
        print(f"Step {step['step']}: Edge {edge} -> {action}")
        if action == 'added':
            print(f"  Visited: {step['visited_after']}")
    print()
    
    # Example 4: Edge cases
    print("8. Edge Cases:")
    
    # Single vertex
    single_vertex = MinimumSpanningTree()
    single_vertex.vertices.add('X')
    single_mst, single_weight, _ = single_vertex.kruskals_mst()
    print(f"Single vertex MST: {single_mst}, Weight: {single_weight}")
    
    # Two vertices
    two_vertex = MinimumSpanningTree()
    two_vertex.add_edge('A', 'B', 5)
    two_mst, two_weight, _ = two_vertex.kruskals_mst()
    print(f"Two vertex MST: {two_mst}, Weight: {two_weight}")
    
    # Verify the MSTs
    print(f"Two vertex MST valid: {two_vertex.verify_mst(two_mst)}")
    
    print("\n=== Demo Complete ===") 