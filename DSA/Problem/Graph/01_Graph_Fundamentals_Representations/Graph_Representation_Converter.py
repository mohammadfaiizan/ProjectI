"""
Graph Representation Converter
Difficulty: Easy

Problem:
Implement a comprehensive graph representation converter that can transform between
different graph representations:
1. Edge List
2. Adjacency Matrix  
3. Adjacency List
4. Compressed Sparse Row (CSR) format

Also implement utility functions for graph analysis and validation.

This is a foundational problem that demonstrates core graph representation concepts.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict

class GraphConverter:
    """
    Complete graph representation converter with analysis utilities
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize converter
        
        Args:
            directed: Whether the graph is directed
        """
        self.directed = directed
    
    def edge_list_to_adjacency_matrix(self, edges: List[List[int]], n: int) -> List[List[int]]:
        """
        Convert edge list to adjacency matrix
        
        Args:
            edges: List of [u, v] or [u, v, weight] edges
            n: Number of vertices
            
        Returns:
            n x n adjacency matrix
            
        Time: O(E + V²)
        Space: O(V²)
        """
        # Initialize matrix with zeros
        matrix = [[0] * n for _ in range(n)]
        
        for edge in edges:
            if len(edge) == 2:  # Unweighted edge
                u, v = edge
                weight = 1
            else:  # Weighted edge
                u, v, weight = edge
            
            matrix[u][v] = weight
            if not self.directed:
                matrix[v][u] = weight
        
        return matrix
    
    def edge_list_to_adjacency_list(self, edges: List[List[int]], n: int) -> Dict[int, List[int]]:
        """
        Convert edge list to adjacency list
        
        Time: O(E + V)
        Space: O(E + V)
        """
        adj_list = defaultdict(list)
        
        # Initialize all vertices
        for i in range(n):
            adj_list[i] = []
        
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
                adj_list[u].append(v)
                if not self.directed:
                    adj_list[v].append(u)
            else:  # Weighted edges
                u, v, weight = edge
                adj_list[u].append((v, weight))
                if not self.directed:
                    adj_list[v].append((u, weight))
        
        return dict(adj_list)
    
    def adjacency_matrix_to_edge_list(self, matrix: List[List[int]]) -> List[List[int]]:
        """
        Convert adjacency matrix to edge list
        
        Time: O(V²)
        Space: O(E)
        """
        edges = []
        n = len(matrix)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] != 0:
                    if self.directed or i <= j:  # Avoid duplicates in undirected
                        if matrix[i][j] == 1:
                            edges.append([i, j])
                        else:
                            edges.append([i, j, matrix[i][j]])
        
        return edges
    
    def adjacency_matrix_to_adjacency_list(self, matrix: List[List[int]]) -> Dict[int, List[int]]:
        """
        Convert adjacency matrix to adjacency list
        
        Time: O(V²)
        Space: O(E + V)
        """
        adj_list = {}
        n = len(matrix)
        
        for i in range(n):
            adj_list[i] = []
            for j in range(n):
                if matrix[i][j] != 0:
                    if matrix[i][j] == 1:
                        adj_list[i].append(j)
                    else:
                        adj_list[i].append((j, matrix[i][j]))
        
        return adj_list
    
    def adjacency_list_to_edge_list(self, adj_list: Dict[int, List]) -> List[List[int]]:
        """
        Convert adjacency list to edge list
        
        Time: O(E)
        Space: O(E)
        """
        edges = []
        visited_edges = set()
        
        for u, neighbors in adj_list.items():
            for neighbor in neighbors:
                if isinstance(neighbor, tuple):  # Weighted
                    v, weight = neighbor
                    edge = (min(u, v), max(u, v), weight) if not self.directed else (u, v, weight)
                else:  # Unweighted
                    v = neighbor
                    edge = (min(u, v), max(u, v)) if not self.directed else (u, v)
                
                if self.directed or edge not in visited_edges:
                    if isinstance(neighbor, tuple):
                        edges.append([u, v, weight])
                    else:
                        edges.append([u, v])
                    visited_edges.add(edge)
        
        return edges
    
    def adjacency_list_to_adjacency_matrix(self, adj_list: Dict[int, List], n: int) -> List[List[int]]:
        """
        Convert adjacency list to adjacency matrix
        
        Time: O(E + V²)
        Space: O(V²)
        """
        matrix = [[0] * n for _ in range(n)]
        
        for u, neighbors in adj_list.items():
            for neighbor in neighbors:
                if isinstance(neighbor, tuple):  # Weighted
                    v, weight = neighbor
                    matrix[u][v] = weight
                else:  # Unweighted
                    v = neighbor
                    matrix[u][v] = 1
        
        return matrix
    
    def to_compressed_sparse_row(self, matrix: List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
        """
        Convert adjacency matrix to Compressed Sparse Row (CSR) format
        
        CSR consists of three arrays:
        - values: non-zero values
        - col_indices: column indices of non-zero values
        - row_pointers: pointers to start of each row in values array
        
        Time: O(V²)
        Space: O(E + V)
        """
        values = []
        col_indices = []
        row_pointers = [0]
        
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val != 0:
                    values.append(val)
                    col_indices.append(j)
            row_pointers.append(len(values))
        
        return values, col_indices, row_pointers
    
    def from_compressed_sparse_row(self, values: List[int], col_indices: List[int], 
                                  row_pointers: List[int], n: int) -> List[List[int]]:
        """
        Convert CSR format back to adjacency matrix
        
        Time: O(E + V²)
        Space: O(V²)
        """
        matrix = [[0] * n for _ in range(n)]
        
        for i in range(len(row_pointers) - 1):
            start = row_pointers[i]
            end = row_pointers[i + 1]
            
            for j in range(start, end):
                col = col_indices[j]
                val = values[j]
                matrix[i][col] = val
        
        return matrix

class GraphAnalyzer:
    """
    Graph analysis utilities
    """
    
    @staticmethod
    def calculate_degrees(adj_list: Dict[int, List]) -> Dict[str, Dict[int, int]]:
        """
        Calculate in-degree and out-degree for each vertex
        """
        vertices = set(adj_list.keys())
        for neighbors in adj_list.values():
            for neighbor in neighbors:
                if isinstance(neighbor, tuple):
                    vertices.add(neighbor[0])
                else:
                    vertices.add(neighbor)
        
        in_degree = {v: 0 for v in vertices}
        out_degree = {v: 0 for v in vertices}
        
        for u, neighbors in adj_list.items():
            out_degree[u] = len(neighbors)
            for neighbor in neighbors:
                v = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                in_degree[v] += 1
        
        return {"in_degree": in_degree, "out_degree": out_degree}
    
    @staticmethod
    def validate_graph_representations(edges: List[List[int]], matrix: List[List[int]], 
                                     adj_list: Dict[int, List]) -> bool:
        """
        Validate that different representations represent the same graph
        """
        # Convert everything to edge lists and compare
        converter_directed = GraphConverter(directed=True)
        converter_undirected = GraphConverter(directed=False)
        
        # Try both directed and undirected interpretations
        for converter in [converter_directed, converter_undirected]:
            edges_from_matrix = converter.adjacency_matrix_to_edge_list(matrix)
            edges_from_adj_list = converter.adjacency_list_to_edge_list(adj_list)
            
            # Normalize edge lists for comparison
            def normalize_edges(edge_list):
                normalized = []
                for edge in edge_list:
                    if len(edge) == 2:
                        normalized.append(tuple(sorted(edge) if not converter.directed else edge))
                    else:
                        normalized.append(tuple(sorted(edge[:2]) + [edge[2]] if not converter.directed else edge))
                return set(normalized)
            
            edges_norm = normalize_edges(edges)
            matrix_norm = normalize_edges(edges_from_matrix)
            adj_list_norm = normalize_edges(edges_from_adj_list)
            
            if edges_norm == matrix_norm == adj_list_norm:
                return True
        
        return False
    
    @staticmethod
    def graph_properties(adj_list: Dict[int, List]) -> Dict[str, any]:
        """
        Calculate basic graph properties
        """
        vertices = set(adj_list.keys())
        edges = 0
        
        for neighbors in adj_list.values():
            edges += len(neighbors)
            for neighbor in neighbors:
                v = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                vertices.add(v)
        
        num_vertices = len(vertices)
        num_edges = edges // 2  # Assuming undirected for this calculation
        
        # Calculate density
        max_edges = num_vertices * (num_vertices - 1) // 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        return {
            "num_vertices": num_vertices,
            "num_edges": num_edges,
            "density": density,
            "is_sparse": density < 0.1,
            "vertices": sorted(vertices)
        }

def test_graph_converter():
    """Test the graph converter with various examples"""
    print("=== Graph Representation Converter Test ===")
    
    # Test case: Simple undirected graph
    edges = [[0, 1], [1, 2], [2, 3], [0, 3]]
    n = 4
    
    print(f"Original edges: {edges}")
    print(f"Number of vertices: {n}")
    
    # Test undirected conversion
    converter = GraphConverter(directed=False)
    
    # Convert to all representations
    adj_matrix = converter.edge_list_to_adjacency_matrix(edges, n)
    adj_list = converter.edge_list_to_adjacency_list(edges, n)
    
    print(f"\nAdjacency Matrix:")
    for i, row in enumerate(adj_matrix):
        print(f"  {i}: {row}")
    
    print(f"\nAdjacency List:")
    for vertex, neighbors in adj_list.items():
        print(f"  {vertex}: {neighbors}")
    
    # Test CSR format
    values, col_indices, row_pointers = converter.to_compressed_sparse_row(adj_matrix)
    print(f"\nCSR Format:")
    print(f"  Values: {values}")
    print(f"  Column indices: {col_indices}")
    print(f"  Row pointers: {row_pointers}")
    
    # Convert back and verify
    recovered_matrix = converter.from_compressed_sparse_row(values, col_indices, row_pointers, n)
    print(f"\nRecovered matrix matches: {adj_matrix == recovered_matrix}")
    
    # Test conversion back to edges
    edges_from_matrix = converter.adjacency_matrix_to_edge_list(adj_matrix)
    edges_from_adj_list = converter.adjacency_list_to_edge_list(adj_list)
    
    print(f"\nEdges from matrix: {edges_from_matrix}")
    print(f"Edges from adj_list: {edges_from_adj_list}")
    
    # Analyze graph properties
    analyzer = GraphAnalyzer()
    properties = analyzer.graph_properties(adj_list)
    degrees = analyzer.calculate_degrees(adj_list)
    
    print(f"\nGraph Properties: {properties}")
    print(f"Degrees: {degrees}")
    
    # Validate representations
    is_valid = analyzer.validate_graph_representations(edges, adj_matrix, adj_list)
    print(f"Representations are consistent: {is_valid}")

def test_weighted_graph():
    """Test with weighted graph"""
    print(f"\n=== Weighted Graph Test ===")
    
    edges = [[0, 1, 5], [1, 2, 3], [2, 0, 7]]
    n = 3
    
    converter = GraphConverter(directed=True)
    
    adj_matrix = converter.edge_list_to_adjacency_matrix(edges, n)
    adj_list = converter.edge_list_to_adjacency_list(edges, n)
    
    print(f"Weighted edges: {edges}")
    print(f"Adjacency Matrix:")
    for i, row in enumerate(adj_matrix):
        print(f"  {i}: {row}")
    
    print(f"Adjacency List:")
    for vertex, neighbors in adj_list.items():
        print(f"  {vertex}: {neighbors}")

if __name__ == "__main__":
    test_graph_converter()
    test_weighted_graph()

"""
Graph Theory Concepts:
1. Graph Representation Trade-offs
2. Space and Time Complexity Analysis
3. Sparse vs Dense Graph Handling
4. Graph Property Analysis

Representation Comparison:
┌─────────────────┬──────────────┬──────────────┬─────────────────┐
│ Representation  │ Space        │ Edge Query   │ Best Use Case   │
├─────────────────┼──────────────┼──────────────┼─────────────────┤
│ Adjacency Matrix│ O(V²)        │ O(1)         │ Dense graphs    │
│ Adjacency List  │ O(V + E)     │ O(degree)    │ Sparse graphs   │
│ Edge List       │ O(E)         │ O(E)         │ Simple storage  │
│ CSR Format      │ O(V + E)     │ O(degree)    │ Sparse + Fast   │
└─────────────────┴──────────────┴──────────────┴─────────────────┘

Real-world Applications:
- Social networks (adjacency list for sparse connections)
- Computer graphics (adjacency matrix for dense meshes)
- Network routing (CSR for efficient sparse matrix operations)
- Database relations (edge list for simple storage)
"""
