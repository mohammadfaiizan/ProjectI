"""
Graph Coloring Basic - Comprehensive Algorithm Implementation
Difficulty: Easy

This file provides comprehensive implementations of basic graph coloring algorithms,
including greedy approaches, backtracking, and optimization techniques.

Key Concepts:
1. Graph Coloring Problem
2. Chromatic Number
3. Greedy Coloring Algorithms
4. Backtracking Approaches
5. Optimization Techniques
6. Applications and Analysis
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import random

class GraphColoringBasic:
    """Comprehensive basic graph coloring algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm performance statistics"""
        self.stats = {
            'colors_used': 0,
            'nodes_processed': 0,
            'backtracks': 0,
            'conflicts_detected': 0,
            'optimizations_applied': 0
        }
    
    def greedy_coloring_first_fit(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 1: Greedy First-Fit Coloring
        
        Color vertices in order using the first available color.
        
        Time: O(V^2)
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'coloring': {}, 'colors_used': 0, 'statistics': self.stats}
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        coloring = {}
        
        for vertex in sorted(vertices):
            self.stats['nodes_processed'] += 1
            
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in graph.get(vertex, []):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[vertex] = color
            self.stats['colors_used'] = max(self.stats['colors_used'], color + 1)
        
        return {
            'coloring': coloring,
            'colors_used': self.stats['colors_used'],
            'is_valid': self._validate_coloring(graph, coloring),
            'statistics': self.stats.copy()
        }
    
    def greedy_coloring_largest_first(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 2: Greedy Largest Degree First Coloring
        
        Color vertices in decreasing order of degree.
        
        Time: O(V^2 + V log V)
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'coloring': {}, 'colors_used': 0, 'statistics': self.stats}
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        # Sort vertices by degree (largest first)
        degree = {v: len(graph.get(v, [])) for v in vertices}
        sorted_vertices = sorted(vertices, key=lambda x: degree[x], reverse=True)
        
        coloring = {}
        
        for vertex in sorted_vertices:
            self.stats['nodes_processed'] += 1
            
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in graph.get(vertex, []):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[vertex] = color
            self.stats['colors_used'] = max(self.stats['colors_used'], color + 1)
        
        return {
            'coloring': coloring,
            'colors_used': self.stats['colors_used'],
            'is_valid': self._validate_coloring(graph, coloring),
            'ordering_strategy': 'largest_degree_first',
            'statistics': self.stats.copy()
        }
    
    def greedy_coloring_smallest_last(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 3: Greedy Smallest Last Coloring
        
        Remove vertices with smallest degree iteratively, then color in reverse order.
        
        Time: O(V^2)
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'coloring': {}, 'colors_used': 0, 'statistics': self.stats}
        
        # Build mutable graph copy
        vertices = set()
        temp_graph = defaultdict(set)
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                temp_graph[u].add(v)
                temp_graph[v].add(u)
        
        removal_order = []
        remaining_vertices = vertices.copy()
        
        # Remove vertices in order of smallest degree
        while remaining_vertices:
            # Find vertex with minimum degree
            min_degree = float('inf')
            min_vertex = None
            
            for v in remaining_vertices:
                degree = len(temp_graph[v] & remaining_vertices)
                if degree < min_degree:
                    min_degree = degree
                    min_vertex = v
            
            # Remove vertex and update graph
            removal_order.append(min_vertex)
            remaining_vertices.remove(min_vertex)
            
            # Update neighbor connections
            for neighbor in temp_graph[min_vertex]:
                temp_graph[neighbor].discard(min_vertex)
            
            self.stats['optimizations_applied'] += 1
        
        # Color in reverse removal order
        coloring = {}
        
        for vertex in reversed(removal_order):
            self.stats['nodes_processed'] += 1
            
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in graph.get(vertex, []):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[vertex] = color
            self.stats['colors_used'] = max(self.stats['colors_used'], color + 1)
        
        return {
            'coloring': coloring,
            'colors_used': self.stats['colors_used'],
            'is_valid': self._validate_coloring(graph, coloring),
            'ordering_strategy': 'smallest_last',
            'statistics': self.stats.copy()
        }
    
    def backtracking_coloring(self, graph: Dict[int, List[int]], max_colors: int = None) -> Dict:
        """
        Approach 4: Backtracking Coloring
        
        Use backtracking to find optimal coloring with minimum colors.
        
        Time: O(k^V) where k is number of colors
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'coloring': {}, 'colors_used': 0, 'statistics': self.stats}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        
        if max_colors is None:
            max_colors = len(vertices)  # Upper bound
        
        coloring = {}
        
        def is_safe(vertex, color):
            """Check if color assignment is safe"""
            for neighbor in graph.get(vertex, []):
                if neighbor in coloring and coloring[neighbor] == color:
                    self.stats['conflicts_detected'] += 1
                    return False
            return True
        
        def backtrack(vertex_idx):
            """Backtracking function"""
            if vertex_idx == len(vertices):
                return True  # All vertices colored
            
            vertex = vertices[vertex_idx]
            self.stats['nodes_processed'] += 1
            
            for color in range(max_colors):
                if is_safe(vertex, color):
                    coloring[vertex] = color
                    
                    if backtrack(vertex_idx + 1):
                        return True
                    
                    # Backtrack
                    del coloring[vertex]
                    self.stats['backtracks'] += 1
            
            return False
        
        # Try to find coloring with decreasing number of colors
        for k in range(1, max_colors + 1):
            coloring.clear()
            self.stats['colors_used'] = k
            
            if backtrack(0):
                return {
                    'coloring': coloring.copy(),
                    'colors_used': k,
                    'is_valid': self._validate_coloring(graph, coloring),
                    'is_optimal': True,
                    'statistics': self.stats.copy()
                }
        
        return {
            'coloring': {},
            'colors_used': 0,
            'is_valid': False,
            'message': 'No valid coloring found',
            'statistics': self.stats.copy()
        }
    
    def dsatur_coloring(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 5: DSATUR Algorithm (Degree of Saturation)
        
        Color vertices based on saturation degree and degree.
        
        Time: O(V^2)
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'coloring': {}, 'colors_used': 0, 'statistics': self.stats}
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        coloring = {}
        uncolored = vertices.copy()
        
        while uncolored:
            # Calculate saturation degree for each uncolored vertex
            best_vertex = None
            max_saturation = -1
            max_degree = -1
            
            for vertex in uncolored:
                # Saturation degree: number of different colors in neighborhood
                neighbor_colors = set()
                for neighbor in graph.get(vertex, []):
                    if neighbor in coloring:
                        neighbor_colors.add(coloring[neighbor])
                
                saturation = len(neighbor_colors)
                degree = len([n for n in graph.get(vertex, []) if n in uncolored])
                
                # Choose vertex with highest saturation, then highest degree
                if (saturation > max_saturation or 
                    (saturation == max_saturation and degree > max_degree)):
                    max_saturation = saturation
                    max_degree = degree
                    best_vertex = vertex
            
            # Color the selected vertex
            vertex = best_vertex
            uncolored.remove(vertex)
            self.stats['nodes_processed'] += 1
            
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in graph.get(vertex, []):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            # Find first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            coloring[vertex] = color
            self.stats['colors_used'] = max(self.stats['colors_used'], color + 1)
        
        return {
            'coloring': coloring,
            'colors_used': self.stats['colors_used'],
            'is_valid': self._validate_coloring(graph, coloring),
            'algorithm': 'DSATUR',
            'statistics': self.stats.copy()
        }
    
    def welsh_powell_coloring(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 6: Welsh-Powell Algorithm
        
        Order vertices by degree and color greedily.
        
        Time: O(V^2 + V log V)
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'coloring': {}, 'colors_used': 0, 'statistics': self.stats}
        
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        # Sort vertices by degree (descending)
        degree = {v: len(graph.get(v, [])) for v in vertices}
        sorted_vertices = sorted(vertices, key=lambda x: degree[x], reverse=True)
        
        coloring = {}
        color = 0
        
        while any(v not in coloring for v in vertices):
            # Color independent set with current color
            colored_with_current = set()
            
            for vertex in sorted_vertices:
                if vertex in coloring:
                    continue
                
                # Check if vertex can be colored with current color
                can_color = True
                for neighbor in graph.get(vertex, []):
                    if neighbor in colored_with_current:
                        can_color = False
                        break
                
                if can_color:
                    coloring[vertex] = color
                    colored_with_current.add(vertex)
                    self.stats['nodes_processed'] += 1
            
            color += 1
            self.stats['colors_used'] = color
        
        return {
            'coloring': coloring,
            'colors_used': self.stats['colors_used'],
            'is_valid': self._validate_coloring(graph, coloring),
            'algorithm': 'Welsh_Powell',
            'statistics': self.stats.copy()
        }
    
    def _validate_coloring(self, graph: Dict[int, List[int]], coloring: Dict[int, int]) -> bool:
        """Validate that coloring is proper"""
        for vertex in graph:
            for neighbor in graph[vertex]:
                if (vertex in coloring and neighbor in coloring and 
                    coloring[vertex] == coloring[neighbor]):
                    return False
        return True

def test_graph_coloring_algorithms():
    """Test all graph coloring algorithms"""
    colorizer = GraphColoringBasic()
    
    print("=== Testing Graph Coloring Algorithms ===")
    
    # Test graphs
    test_graphs = [
        # Simple path
        {0: [1], 1: [0, 2], 2: [1]},
        
        # Triangle (3-clique)
        {0: [1, 2], 1: [0, 2], 2: [0, 1]},
        
        # 4-cycle
        {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]},
        
        # Complete graph K4
        {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]},
        
        # Bipartite graph
        {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]},
        
        # Star graph
        {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]},
        
        # Empty graph
        {0: [], 1: [], 2: []},
    ]
    
    algorithms = [
        ("Greedy First-Fit", colorizer.greedy_coloring_first_fit),
        ("Largest Degree First", colorizer.greedy_coloring_largest_first),
        ("Smallest Last", colorizer.greedy_coloring_smallest_last),
        ("DSATUR", colorizer.dsatur_coloring),
        ("Welsh-Powell", colorizer.welsh_powell_coloring),
    ]
    
    for i, graph in enumerate(test_graphs):
        print(f"\n--- Test Graph {i+1} ---")
        print(f"Graph: {graph}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                colors = result['colors_used']
                valid = result['is_valid']
                nodes = result['statistics']['nodes_processed']
                
                status = "✓" if valid else "✗"
                print(f"{alg_name:20} | {status} | Colors: {colors:2} | Nodes: {nodes:3}")
                
            except Exception as e:
                print(f"{alg_name:20} | ERROR: {str(e)[:30]}")

def demonstrate_coloring_process():
    """Demonstrate step-by-step coloring process"""
    print("\n=== Graph Coloring Process Demo ===")
    
    graph = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
    
    print(f"Graph: {graph}")
    print(f"Structure: Square with vertices 0-1-2-3-0")
    
    colorizer = GraphColoringBasic()
    
    print(f"\n=== Greedy First-Fit Process ===")
    result = colorizer.greedy_coloring_first_fit(graph)
    coloring = result['coloring']
    
    print(f"Vertex processing order: {sorted(graph.keys())}")
    for vertex in sorted(graph.keys()):
        neighbors = graph[vertex]
        neighbor_colors = [coloring.get(n, 'uncolored') for n in neighbors]
        print(f"Vertex {vertex}: neighbors={neighbors}, neighbor_colors={neighbor_colors}, assigned_color={coloring[vertex]}")
    
    print(f"\nFinal coloring: {coloring}")
    print(f"Colors used: {result['colors_used']}")
    print(f"Valid: {result['is_valid']}")

def demonstrate_algorithm_comparison():
    """Demonstrate comparison between different algorithms"""
    print("\n=== Algorithm Comparison Demo ===")
    
    # Create test graph with known chromatic number
    graph = {
        0: [1, 3, 4],
        1: [0, 2, 4],
        2: [1, 3, 4],
        3: [0, 2, 4],
        4: [0, 1, 2, 3]
    }
    
    print(f"Test graph (wheel W4): {graph}")
    print(f"Known chromatic number: 4 (complete subgraph K4)")
    
    colorizer = GraphColoringBasic()
    
    algorithms = [
        ("Greedy First-Fit", colorizer.greedy_coloring_first_fit),
        ("Largest Degree First", colorizer.greedy_coloring_largest_first),
        ("Smallest Last", colorizer.greedy_coloring_smallest_last),
        ("DSATUR", colorizer.dsatur_coloring),
        ("Welsh-Powell", colorizer.welsh_powell_coloring),
    ]
    
    print(f"\nAlgorithm performance:")
    print(f"{'Algorithm':<20} | {'Colors':<6} | {'Valid':<5} | {'Nodes':<5}")
    print("-" * 50)
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(graph)
            colors = result['colors_used']
            valid = "Yes" if result['is_valid'] else "No"
            nodes = result['statistics']['nodes_processed']
            
            print(f"{alg_name:<20} | {colors:<6} | {valid:<5} | {nodes:<5}")
            
        except Exception as e:
            print(f"{alg_name:<20} | ERROR: {str(e)[:20]}")

def analyze_coloring_theory():
    """Analyze graph coloring theory and bounds"""
    print("\n=== Graph Coloring Theory ===")
    
    print("Theoretical Foundations:")
    
    print("\n1. **Chromatic Number χ(G):**")
    print("   • Minimum colors needed for proper coloring")
    print("   • χ(G) = 1 ⟺ G has no edges")
    print("   • χ(G) = 2 ⟺ G is bipartite and connected")
    print("   • χ(G) = |V| ⟺ G is complete graph")
    
    print("\n2. **Bounds and Results:**")
    print("   • ω(G) ≤ χ(G) ≤ Δ(G) + 1")
    print("   • ω(G) = clique number, Δ(G) = maximum degree")
    print("   • Brooks' theorem: χ(G) ≤ Δ(G) unless complete or odd cycle")
    print("   • Four color theorem: χ(G) ≤ 4 for planar graphs")
    
    print("\n3. **Algorithm Performance:**")
    print("   • Greedy: χ(G) ≤ Δ(G) + 1 guarantee")
    print("   • DSATUR: Often better than greedy in practice")
    print("   • Welsh-Powell: Good for sparse graphs")
    print("   • Backtracking: Optimal but exponential time")
    
    print("\n4. **Complexity Results:**")
    print("   • Graph coloring is NP-complete")
    print("   • k-coloring is polynomial for k=2, NP-complete for k≥3")
    print("   • No polynomial approximation better than n^(1-ε)")
    print("   • Heuristics work well in practice")
    
    print("\n5. **Special Graph Classes:**")
    print("   • Bipartite graphs: χ(G) = 2")
    print("   • Trees: χ(G) = 2 (unless single vertex)")
    print("   • Planar graphs: χ(G) ≤ 4")
    print("   • Perfect graphs: χ(G) = ω(G)")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of graph coloring"""
    print("\n=== Real-World Applications ===")
    
    print("Graph Coloring Applications:")
    
    print("\n1. **Scheduling Problems:**")
    print("   • Course scheduling (no time conflicts)")
    print("   • Exam scheduling")
    print("   • Meeting room assignment")
    print("   • Task scheduling with resource conflicts")
    
    print("\n2. **Register Allocation:**")
    print("   • Compiler optimization")
    print("   • Variable assignment to CPU registers")
    print("   • Minimize memory usage")
    print("   • Interference graph coloring")
    
    print("\n3. **Frequency Assignment:**")
    print("   • Radio frequency allocation")
    print("   • Cell tower frequency planning")
    print("   • Avoid interference between stations")
    print("   • Optimize spectrum usage")
    
    print("\n4. **Map Coloring:**")
    print("   • Geographic map coloring")
    print("   • Political boundary visualization")
    print("   • Minimize color usage")
    print("   • Four color theorem application")
    
    print("\n5. **Network Design:**")
    print("   • TDMA slot assignment")
    print("   • Wavelength assignment in optical networks")
    print("   • Channel assignment in wireless networks")
    print("   • Conflict-free resource allocation")

def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques for graph coloring"""
    print("\n=== Optimization Techniques ===")
    
    print("Graph Coloring Optimizations:")
    
    print("\n1. **Vertex Ordering Strategies:**")
    print("   • Largest degree first: High-degree vertices first")
    print("   • Smallest last: Remove low-degree iteratively")
    print("   • Random ordering: Avoid worst-case behavior")
    print("   • Saturation degree (DSATUR): Dynamic ordering")
    
    print("\n2. **Pruning Techniques:**")
    print("   • Early termination on infeasible colorings")
    print("   • Bound propagation in backtracking")
    print("   • Symmetry breaking in search")
    print("   • Conflict-driven learning")
    
    print("\n3. **Heuristic Improvements:**")
    print("   • Local search and improvement")
    print("   • Simulated annealing")
    print("   • Genetic algorithms")
    print("   • Hybrid approaches")
    
    print("\n4. **Data Structure Optimizations:**")
    print("   • Efficient neighbor lookup")
    print("   • Bit manipulation for color sets")
    print("   • Cache-friendly graph representation")
    print("   • Parallel processing techniques")
    
    print("\n5. **Problem-Specific Optimizations:**")
    print("   • Exploit special graph structure")
    print("   • Precompute cliques and independent sets")
    print("   • Use problem domain knowledge")
    print("   • Approximate solutions for large instances")

if __name__ == "__main__":
    test_graph_coloring_algorithms()
    demonstrate_coloring_process()
    demonstrate_algorithm_comparison()
    analyze_coloring_theory()
    demonstrate_real_world_applications()
    demonstrate_optimization_techniques()

"""
Graph Coloring and Algorithm Theory Concepts:
1. Fundamental Graph Coloring Algorithms and Heuristics
2. Greedy Approaches and Vertex Ordering Strategies
3. Backtracking and Optimization Techniques
4. Theoretical Bounds and Complexity Analysis
5. Real-world Applications in Scheduling and Resource Allocation

Key Algorithmic Insights:
- Multiple greedy strategies with different performance characteristics
- Vertex ordering significantly impacts solution quality
- DSATUR often provides best practical results
- Backtracking finds optimal solutions but is exponential
- Problem is NP-complete but heuristics work well in practice

Algorithm Strategy:
1. Choose appropriate vertex ordering strategy
2. Apply greedy coloring with conflict checking
3. Use advanced heuristics for better solutions
4. Apply optimization techniques for large instances

Theoretical Foundations:
- Chromatic number bounds and relationships
- Brooks' theorem and structural results
- Four color theorem for planar graphs
- Perfect graph theory and special cases
- Complexity theory and approximation hardness

Optimization Techniques:
- Sophisticated vertex ordering (DSATUR, smallest-last)
- Backtracking with pruning and bounds
- Local search and metaheuristic approaches
- Data structure and implementation optimizations

Real-world Applications:
- Course and exam scheduling optimization
- Compiler register allocation
- Frequency assignment in wireless networks
- Map coloring and visualization
- Resource allocation with conflict constraints

This comprehensive implementation provides fundamental
graph coloring algorithms essential for optimization problems.
"""
