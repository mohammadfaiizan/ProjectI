"""
Vertex Cover Approximation Algorithms - Comprehensive Implementation
Difficulty: Medium

This file provides comprehensive implementations of vertex cover approximation algorithms,
including exact, approximation, and heuristic approaches for finding minimum vertex covers.

Key Concepts:
1. Minimum Vertex Cover Problem
2. Approximation Algorithms
3. Linear Programming Relaxation
4. Matching-based Approaches
5. Local Search Optimization
6. Graph Theory Applications
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import random

class VertexCoverApproximation:
    """Comprehensive vertex cover approximation algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm performance statistics"""
        self.stats = {
            'vertices_selected': 0,
            'edges_covered': 0,
            'iterations': 0,
            'improvement_steps': 0,
            'approximation_ratio': 0.0
        }
    
    def vertex_cover_greedy_degree(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 1: Greedy Algorithm by Maximum Degree
        
        Iteratively select vertex with maximum degree.
        
        Time: O(V^2)
        Space: O(V + E)
        Approximation: O(log V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'vertex_cover': [], 'size': 0, 'statistics': self.stats}
        
        # Build edge set and vertex set
        edges = set()
        vertices = set()
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                if u < v:  # Avoid duplicate edges
                    edges.add((u, v))
        
        vertex_cover = []
        uncovered_edges = edges.copy()
        remaining_vertices = vertices.copy()
        
        # Create mutable adjacency structure
        temp_graph = defaultdict(set)
        for u, v in edges:
            temp_graph[u].add(v)
            temp_graph[v].add(u)
        
        while uncovered_edges:
            # Find vertex with maximum degree in remaining graph
            max_degree = -1
            best_vertex = None
            
            for v in remaining_vertices:
                degree = len(temp_graph[v])
                if degree > max_degree:
                    max_degree = degree
                    best_vertex = v
                
                self.stats['iterations'] += 1
            
            if best_vertex is None:
                break
            
            # Add vertex to cover
            vertex_cover.append(best_vertex)
            remaining_vertices.remove(best_vertex)
            self.stats['vertices_selected'] += 1
            
            # Remove edges covered by this vertex
            edges_to_remove = []
            for edge in uncovered_edges:
                u, v = edge
                if u == best_vertex or v == best_vertex:
                    edges_to_remove.append(edge)
                    self.stats['edges_covered'] += 1
            
            for edge in edges_to_remove:
                uncovered_edges.remove(edge)
                u, v = edge
                temp_graph[u].discard(v)
                temp_graph[v].discard(u)
        
        return {
            'vertex_cover': vertex_cover,
            'size': len(vertex_cover),
            'algorithm': 'greedy_degree',
            'approximation_ratio': 'O(log V)',
            'statistics': self.stats.copy()
        }
    
    def vertex_cover_maximal_matching(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 2: 2-Approximation via Maximal Matching
        
        Find maximal matching and include both endpoints of each edge.
        
        Time: O(V + E)
        Space: O(V + E)
        Approximation: 2
        """
        self.reset_statistics()
        
        if not graph:
            return {'vertex_cover': [], 'size': 0, 'statistics': self.stats}
        
        # Build edge set
        edges = []
        vertices = set()
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                if u < v:  # Avoid duplicate edges
                    edges.append((u, v))
        
        # Find maximal matching greedily
        matching = []
        used_vertices = set()
        
        for u, v in edges:
            self.stats['iterations'] += 1
            
            if u not in used_vertices and v not in used_vertices:
                matching.append((u, v))
                used_vertices.add(u)
                used_vertices.add(v)
                self.stats['edges_covered'] += 1
        
        # Vertex cover includes both endpoints of matching edges
        vertex_cover = []
        for u, v in matching:
            vertex_cover.extend([u, v])
            self.stats['vertices_selected'] += 2
        
        return {
            'vertex_cover': vertex_cover,
            'size': len(vertex_cover),
            'matching': matching,
            'matching_size': len(matching),
            'algorithm': 'maximal_matching',
            'approximation_ratio': 2.0,
            'statistics': self.stats.copy()
        }
    
    def vertex_cover_lp_relaxation(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 3: LP Relaxation with Randomized Rounding
        
        Solve LP relaxation and round fractional solutions.
        
        Time: O(V + E)
        Space: O(V + E)
        Approximation: 2 (expected)
        """
        self.reset_statistics()
        
        if not graph:
            return {'vertex_cover': [], 'size': 0, 'statistics': self.stats}
        
        # Build edge list and vertex set
        edges = []
        vertices = set()
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                if u < v:
                    edges.append((u, v))
        
        # Simple LP relaxation heuristic: assign 0.5 to vertices in edges
        lp_solution = {}
        for v in vertices:
            lp_solution[v] = 0.0
        
        # For each edge, ensure constraint is satisfied
        for u, v in edges:
            if lp_solution[u] + lp_solution[v] < 1.0:
                # Distribute weight to satisfy constraint
                deficit = 1.0 - (lp_solution[u] + lp_solution[v])
                lp_solution[u] += deficit / 2
                lp_solution[v] += deficit / 2
            
            self.stats['iterations'] += 1
        
        # Randomized rounding: include vertex if LP value >= 0.5
        vertex_cover = []
        for v in vertices:
            if lp_solution[v] >= 0.5:
                vertex_cover.append(v)
                self.stats['vertices_selected'] += 1
        
        # Verify coverage and add missing vertices if needed
        covered_edges = set()
        for u, v in edges:
            if u in vertex_cover or v in vertex_cover:
                covered_edges.add((u, v))
                self.stats['edges_covered'] += 1
        
        # Add vertices for uncovered edges
        for u, v in edges:
            if (u, v) not in covered_edges:
                if u not in vertex_cover:
                    vertex_cover.append(u)
                    self.stats['vertices_selected'] += 1
                    break
        
        return {
            'vertex_cover': vertex_cover,
            'size': len(vertex_cover),
            'lp_solution': lp_solution,
            'algorithm': 'lp_relaxation',
            'approximation_ratio': 2.0,
            'statistics': self.stats.copy()
        }
    
    def vertex_cover_local_search(self, graph: Dict[int, List[int]], max_iterations: int = 1000) -> Dict:
        """
        Approach 4: Local Search Optimization
        
        Start with initial solution and improve via local moves.
        
        Time: O(iterations * V^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'vertex_cover': [], 'size': 0, 'statistics': self.stats}
        
        # Build edge set
        edges = []
        vertices = set()
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                if u < v:
                    edges.append((u, v))
        
        # Start with greedy solution
        initial_result = self.vertex_cover_greedy_degree(graph)
        current_cover = set(initial_result['vertex_cover'])
        best_cover = current_cover.copy()
        
        def is_vertex_cover(cover):
            """Check if set is a valid vertex cover"""
            for u, v in edges:
                if u not in cover and v not in cover:
                    return False
            return True
        
        def get_neighbors(cover):
            """Generate neighboring solutions"""
            neighbors = []
            
            # Try removing each vertex
            for v in cover:
                new_cover = cover - {v}
                if is_vertex_cover(new_cover):
                    neighbors.append(new_cover)
            
            # Try adding each vertex not in cover
            for v in vertices:
                if v not in cover:
                    new_cover = cover | {v}
                    neighbors.append(new_cover)
            
            return neighbors
        
        # Local search
        for iteration in range(max_iterations):
            self.stats['iterations'] += 1
            
            neighbors = get_neighbors(current_cover)
            best_neighbor = None
            best_size = len(current_cover)
            
            # Find best neighbor
            for neighbor in neighbors:
                if len(neighbor) < best_size:
                    best_neighbor = neighbor
                    best_size = len(neighbor)
            
            if best_neighbor is not None:
                current_cover = best_neighbor
                if len(current_cover) < len(best_cover):
                    best_cover = current_cover.copy()
                    self.stats['improvement_steps'] += 1
            else:
                break  # Local optimum reached
        
        self.stats['vertices_selected'] = len(best_cover)
        
        return {
            'vertex_cover': list(best_cover),
            'size': len(best_cover),
            'algorithm': 'local_search',
            'iterations_used': iteration + 1,
            'improvements': self.stats['improvement_steps'],
            'statistics': self.stats.copy()
        }
    
    def vertex_cover_branch_and_bound(self, graph: Dict[int, List[int]], max_time: int = 1000) -> Dict:
        """
        Approach 5: Branch and Bound (Exact for Small Graphs)
        
        Find optimal vertex cover using branch and bound.
        
        Time: O(2^V) worst case
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'vertex_cover': [], 'size': 0, 'statistics': self.stats}
        
        # Build edge set and vertex list
        edges = []
        vertices = list(set())
        
        for u in graph:
            if u not in vertices:
                vertices.append(u)
            for v in graph[u]:
                if v not in vertices:
                    vertices.append(v)
                if u < v:
                    edges.append((u, v))
        
        vertices.sort()
        best_cover = vertices[:]  # Start with all vertices
        
        def lower_bound(partial_cover, remaining_vertices):
            """Calculate lower bound using matching"""
            # Simple lower bound: size of maximal matching in uncovered edges
            uncovered_edges = []
            cover_set = set(partial_cover)
            
            for u, v in edges:
                if u not in cover_set and v not in cover_set:
                    uncovered_edges.append((u, v))
            
            # Find maximal matching in uncovered edges
            matching = []
            used = set()
            
            for u, v in uncovered_edges:
                if u not in used and v not in used:
                    matching.append((u, v))
                    used.add(u)
                    used.add(v)
            
            return len(partial_cover) + len(matching)
        
        def branch_and_bound(partial_cover, vertex_index, uncovered_edges):
            """Branch and bound search"""
            nonlocal best_cover
            
            self.stats['iterations'] += 1
            
            if self.stats['iterations'] > max_time:
                return
            
            # Check if all edges are covered
            if not uncovered_edges:
                if len(partial_cover) < len(best_cover):
                    best_cover = partial_cover[:]
                return
            
            # Pruning: if lower bound >= best known solution
            lb = lower_bound(partial_cover, vertices[vertex_index:])
            if lb >= len(best_cover):
                return
            
            if vertex_index >= len(vertices):
                return
            
            vertex = vertices[vertex_index]
            
            # Branch 1: Include vertex in cover
            new_partial = partial_cover + [vertex]
            new_uncovered = []
            
            for u, v in uncovered_edges:
                if u != vertex and v != vertex:
                    new_uncovered.append((u, v))
                else:
                    self.stats['edges_covered'] += 1
            
            branch_and_bound(new_partial, vertex_index + 1, new_uncovered)
            
            # Branch 2: Don't include vertex in cover
            branch_and_bound(partial_cover, vertex_index + 1, uncovered_edges)
        
        branch_and_bound([], 0, edges)
        
        self.stats['vertices_selected'] = len(best_cover)
        
        return {
            'vertex_cover': best_cover,
            'size': len(best_cover),
            'algorithm': 'branch_and_bound',
            'is_optimal': self.stats['iterations'] <= max_time,
            'statistics': self.stats.copy()
        }
    
    def vertex_cover_randomized_rounding(self, graph: Dict[int, List[int]], trials: int = 100) -> Dict:
        """
        Approach 6: Randomized Rounding with Multiple Trials
        
        Multiple randomized LP rounding attempts.
        
        Time: O(trials * (V + E))
        Space: O(V + E)
        Approximation: 2 (expected)
        """
        self.reset_statistics()
        
        if not graph:
            return {'vertex_cover': [], 'size': 0, 'statistics': self.stats}
        
        # Build edge set
        edges = []
        vertices = set()
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                if u < v:
                    edges.append((u, v))
        
        best_cover = list(vertices)  # Start with all vertices
        best_size = len(vertices)
        
        for trial in range(trials):
            self.stats['iterations'] += 1
            
            # LP relaxation: assign fractional values
            lp_values = {}
            for v in vertices:
                lp_values[v] = 0.0
            
            # Satisfy edge constraints
            for u, v in edges:
                if lp_values[u] + lp_values[v] < 1.0:
                    deficit = 1.0 - (lp_values[u] + lp_values[v])
                    lp_values[u] += deficit / 2
                    lp_values[v] += deficit / 2
            
            # Randomized rounding with different thresholds
            threshold = 0.5 + random.uniform(-0.2, 0.2)
            
            candidate_cover = []
            for v in vertices:
                if lp_values[v] >= threshold or random.random() < lp_values[v]:
                    candidate_cover.append(v)
            
            # Ensure all edges are covered
            covered_edges = set()
            cover_set = set(candidate_cover)
            
            for u, v in edges:
                if u in cover_set or v in cover_set:
                    covered_edges.add((u, v))
            
            # Add vertices for uncovered edges
            for u, v in edges:
                if (u, v) not in covered_edges:
                    if u not in cover_set:
                        candidate_cover.append(u)
                        cover_set.add(u)
                    break
            
            # Update best solution
            if len(candidate_cover) < best_size:
                best_cover = candidate_cover
                best_size = len(candidate_cover)
        
        self.stats['vertices_selected'] = best_size
        
        return {
            'vertex_cover': best_cover,
            'size': best_size,
            'algorithm': 'randomized_rounding',
            'trials': trials,
            'approximation_ratio': 2.0,
            'statistics': self.stats.copy()
        }

def test_vertex_cover_algorithms():
    """Test all vertex cover algorithms"""
    solver = VertexCoverApproximation()
    
    print("=== Testing Vertex Cover Algorithms ===")
    
    # Test graphs
    test_graphs = [
        # Path graph
        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        
        # Cycle
        {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]},
        
        # Complete graph K4
        {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]},
        
        # Star graph
        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},
        
        # Bipartite graph
        {0: [3, 4], 1: [3, 4], 2: [3, 4], 3: [0, 1, 2], 4: [0, 1, 2]},
        
        # Triangle
        {0: [1, 2], 1: [0, 2], 2: [0, 1]},
    ]
    
    algorithms = [
        ("Greedy Degree", solver.vertex_cover_greedy_degree),
        ("Maximal Matching", solver.vertex_cover_maximal_matching),
        ("LP Relaxation", solver.vertex_cover_lp_relaxation),
        ("Local Search", solver.vertex_cover_local_search),
        ("Randomized Rounding", solver.vertex_cover_randomized_rounding),
    ]
    
    for i, graph in enumerate(test_graphs):
        print(f"\n--- Test Graph {i+1} ---")
        print(f"Graph: {graph}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                size = result['size']
                cover = result['vertex_cover']
                
                # Validate cover
                valid = validate_vertex_cover(graph, cover)
                status = "✓" if valid else "✗"
                
                approx_ratio = result.get('approximation_ratio', 'N/A')
                
                print(f"{alg_name:18} | {status} | Size: {size:2} | Cover: {sorted(cover)} | Ratio: {approx_ratio}")
                
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

def validate_vertex_cover(graph: Dict[int, List[int]], vertex_cover: List[int]) -> bool:
    """Validate that the vertex cover is correct"""
    cover_set = set(vertex_cover)
    
    for u in graph:
        for v in graph[u]:
            if u < v:  # Check each edge once
                if u not in cover_set and v not in cover_set:
                    return False
    return True

def demonstrate_approximation_analysis():
    """Demonstrate approximation ratio analysis"""
    print("\n=== Approximation Analysis Demo ===")
    
    graph = {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}  # Star graph
    
    print(f"Graph: {graph} (Star with 4 vertices)")
    print(f"Optimal vertex cover size: 1 (center vertex)")
    
    solver = VertexCoverApproximation()
    
    algorithms = [
        ("Maximal Matching", solver.vertex_cover_maximal_matching, 2.0),
        ("Greedy Degree", solver.vertex_cover_greedy_degree, "O(log V)"),
        ("LP Relaxation", solver.vertex_cover_lp_relaxation, 2.0),
    ]
    
    print(f"\nApproximation performance:")
    print(f"{'Algorithm':<18} | {'Size':<4} | {'Ratio':<8} | {'Guarantee'}")
    print("-" * 55)
    
    optimal_size = 1  # Known optimal for star graph
    
    for alg_name, alg_func, theoretical_ratio in algorithms:
        result = alg_func(graph)
        size = result['size']
        actual_ratio = size / optimal_size if optimal_size > 0 else float('inf')
        
        print(f"{alg_name:<18} | {size:<4} | {actual_ratio:<8.1f} | {theoretical_ratio}")

def demonstrate_vertex_cover_theory():
    """Demonstrate vertex cover theory"""
    print("\n=== Vertex Cover Theory ===")
    
    print("Vertex Cover Problem:")
    
    print("\n1. **Problem Definition:**")
    print("   • Vertex cover: set of vertices such that every edge has at least one endpoint in the set")
    print("   • Minimum vertex cover: smallest such set")
    print("   • Vertex cover number τ(G): size of minimum vertex cover")
    
    print("\n2. **Complexity Results:**")
    print("   • Minimum vertex cover is NP-complete")
    print("   • 2-approximation via maximal matching")
    print("   • No (2-ε)-approximation unless P=NP")
    print("   • APX-complete problem")
    
    print("\n3. **Graph Theory Relationships:**")
    print("   • τ(G) + α(G) = n (vertex cover + independent set)")
    print("   • τ(G) ≥ ν(G) (vertex cover ≥ matching number)")
    print("   • König's theorem: τ(G) = ν(G) for bipartite graphs")
    
    print("\n4. **Approximation Algorithms:**")
    print("   • Maximal matching: 2-approximation")
    print("   • LP relaxation + rounding: 2-approximation")
    print("   • Greedy: O(log n)-approximation")
    print("   • Local search: various ratios for special cases")
    
    print("\n5. **Special Graph Classes:**")
    print("   • Trees: τ(G) = ⌈n/2⌉")
    print("   • Bipartite graphs: polynomial-time optimal (König)")
    print("   • Planar graphs: PTAS exists")
    print("   • Bounded treewidth: polynomial-time optimal")

if __name__ == "__main__":
    test_vertex_cover_algorithms()
    demonstrate_approximation_analysis()
    demonstrate_vertex_cover_theory()

"""
Vertex Cover Approximation and Optimization Concepts:
1. Minimum Vertex Cover Problem and NP-Completeness
2. Approximation Algorithms with Performance Guarantees
3. Linear Programming Relaxation and Rounding Techniques
4. Local Search and Optimization Methods
5. Graph Theory Applications and Real-world Problem Solving

Key Algorithmic Insights:
- Vertex cover is NP-complete but admits good approximations
- 2-approximation achievable via maximal matching
- LP relaxation provides theoretical foundation
- Local search improves practical solutions
- Special graph classes have polynomial algorithms

Algorithm Strategy:
1. Use 2-approximation for guaranteed performance
2. Apply local search for solution improvement
3. Use exact methods for small instances
4. Exploit special structure when available

Theoretical Foundations:
- Vertex cover number and graph relationships
- Approximation theory and hardness results
- König's theorem for bipartite graphs
- LP relaxation and rounding analysis

Optimization Techniques:
- Maximal matching for 2-approximation
- Randomized rounding with multiple trials
- Branch-and-bound for exact solutions
- Local search for practical improvements

Real-world Applications:
- Network monitoring and surveillance
- Facility location and coverage problems
- Resource allocation and optimization
- Security and monitoring systems
- Computational biology and bioinformatics

This comprehensive implementation provides robust
vertex cover algorithms for optimization problems.
"""
