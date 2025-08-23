"""
Independent Set Algorithms - Comprehensive Implementation
Difficulty: Medium

This file provides comprehensive implementations of independent set algorithms,
including exact, approximation, and heuristic approaches for finding maximum independent sets.

Key Concepts:
1. Maximum Independent Set Problem
2. Approximation Algorithms
3. Greedy Heuristics
4. Backtracking and Branch-and-Bound
5. Graph Theory Applications
6. Complement Graph Analysis
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import random

class IndependentSetAlgorithms:
    """Comprehensive independent set algorithm implementations"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm performance statistics"""
        self.stats = {
            'nodes_explored': 0,
            'branches_pruned': 0,
            'independent_sets_found': 0,
            'max_size_found': 0,
            'iterations': 0
        }
    
    def maximum_independent_set_greedy_degree(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 1: Greedy Algorithm by Minimum Degree
        
        Iteratively select vertex with minimum degree and remove its neighbors.
        
        Time: O(V^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'independent_set': [], 'size': 0, 'statistics': self.stats}
        
        # Create mutable copy of graph
        vertices = set()
        temp_graph = defaultdict(set)
        
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
                temp_graph[u].add(v)
                temp_graph[v].add(u)
        
        independent_set = []
        remaining_vertices = vertices.copy()
        
        while remaining_vertices:
            # Find vertex with minimum degree in remaining graph
            min_degree = float('inf')
            min_vertex = None
            
            for v in remaining_vertices:
                degree = len(temp_graph[v] & remaining_vertices)
                if degree < min_degree:
                    min_degree = degree
                    min_vertex = v
                
                self.stats['iterations'] += 1
            
            # Add minimum degree vertex to independent set
            independent_set.append(min_vertex)
            remaining_vertices.remove(min_vertex)
            self.stats['nodes_explored'] += 1
            
            # Remove all neighbors of selected vertex
            neighbors_to_remove = temp_graph[min_vertex] & remaining_vertices
            for neighbor in neighbors_to_remove:
                remaining_vertices.remove(neighbor)
                self.stats['branches_pruned'] += 1
        
        self.stats['max_size_found'] = len(independent_set)
        self.stats['independent_sets_found'] = 1
        
        return {
            'independent_set': independent_set,
            'size': len(independent_set),
            'is_maximal': True,
            'algorithm': 'greedy_minimum_degree',
            'statistics': self.stats.copy()
        }
    
    def maximum_independent_set_backtracking(self, graph: Dict[int, List[int]], max_iterations: int = 10000) -> Dict:
        """
        Approach 2: Backtracking with Pruning
        
        Use backtracking to find optimal independent set with branch pruning.
        
        Time: O(2^V) worst case, much better with pruning
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'independent_set': [], 'size': 0, 'statistics': self.stats}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        vertices.sort()  # Consistent ordering
        
        best_independent_set = []
        current_set = []
        
        def can_add_vertex(vertex, current_set):
            """Check if vertex can be added to current independent set"""
            for v in current_set:
                if v in graph.get(vertex, []) or vertex in graph.get(v, []):
                    return False
            return True
        
        def backtrack(index):
            """Backtracking function with pruning"""
            nonlocal best_independent_set
            
            self.stats['iterations'] += 1
            if self.stats['iterations'] > max_iterations:
                return
            
            if index == len(vertices):
                if len(current_set) > len(best_independent_set):
                    best_independent_set = current_set[:]
                    self.stats['independent_sets_found'] += 1
                    self.stats['max_size_found'] = len(best_independent_set)
                return
            
            # Pruning: if current set + remaining vertices can't beat best, skip
            if len(current_set) + (len(vertices) - index) <= len(best_independent_set):
                self.stats['branches_pruned'] += 1
                return
            
            vertex = vertices[index]
            self.stats['nodes_explored'] += 1
            
            # Try including current vertex
            if can_add_vertex(vertex, current_set):
                current_set.append(vertex)
                backtrack(index + 1)
                current_set.pop()
            
            # Try excluding current vertex
            backtrack(index + 1)
        
        backtrack(0)
        
        return {
            'independent_set': best_independent_set,
            'size': len(best_independent_set),
            'is_optimal': self.stats['iterations'] <= max_iterations,
            'algorithm': 'backtracking',
            'statistics': self.stats.copy()
        }
    
    def maximum_independent_set_complement_clique(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 3: Complement Graph Maximum Clique
        
        Find maximum clique in complement graph (equivalent to max independent set).
        
        Time: O(V^3) for clique finding heuristic
        Space: O(V^2)
        """
        self.reset_statistics()
        
        if not graph:
            return {'independent_set': [], 'size': 0, 'statistics': self.stats}
        
        # Find all vertices
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        # Build complement graph
        complement = defaultdict(set)
        for u in vertices:
            for v in vertices:
                if u != v and v not in graph.get(u, []):
                    complement[u].add(v)
        
        # Find maximum clique in complement graph using greedy heuristic
        def greedy_clique():
            """Greedy algorithm to find large clique"""
            clique = []
            candidates = set(vertices)
            
            while candidates:
                # Choose vertex with maximum connections to current candidates
                best_vertex = None
                max_connections = -1
                
                for v in candidates:
                    connections = len(complement[v] & candidates)
                    if connections > max_connections:
                        max_connections = connections
                        best_vertex = v
                    
                    self.stats['iterations'] += 1
                
                if best_vertex is None:
                    break
                
                # Add vertex to clique
                clique.append(best_vertex)
                candidates.remove(best_vertex)
                self.stats['nodes_explored'] += 1
                
                # Update candidates to vertices connected to all clique members
                new_candidates = set()
                for v in candidates:
                    if all(v in complement[c] for c in clique):
                        new_candidates.add(v)
                    else:
                        self.stats['branches_pruned'] += 1
                
                candidates = new_candidates
            
            return clique
        
        clique = greedy_clique()
        
        self.stats['max_size_found'] = len(clique)
        self.stats['independent_sets_found'] = 1
        
        return {
            'independent_set': clique,
            'size': len(clique),
            'is_maximal': True,
            'algorithm': 'complement_clique',
            'statistics': self.stats.copy()
        }
    
    def maximum_independent_set_branch_and_bound(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 4: Branch and Bound with Sophisticated Bounds
        
        Use branch and bound with upper bound estimation for pruning.
        
        Time: O(2^V) worst case, efficient with good bounds
        Space: O(V)
        """
        self.reset_statistics()
        
        if not graph:
            return {'independent_set': [], 'size': 0, 'statistics': self.stats}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        
        best_solution = []
        
        def upper_bound(remaining_vertices, current_size):
            """Calculate upper bound using fractional independent set"""
            if not remaining_vertices:
                return current_size
            
            # Greedy fractional relaxation
            temp_vertices = list(remaining_vertices)
            temp_vertices.sort(key=lambda x: len(graph.get(x, [])))  # Sort by degree
            
            bound = current_size
            remaining = set(temp_vertices)
            
            for v in temp_vertices:
                if v in remaining:
                    bound += 1
                    # Remove neighbors
                    for neighbor in graph.get(v, []):
                        remaining.discard(neighbor)
            
            return bound
        
        def branch_and_bound(remaining_vertices, current_set, forbidden):
            """Branch and bound search"""
            nonlocal best_solution
            
            self.stats['iterations'] += 1
            
            # Check if current solution is better
            if len(current_set) > len(best_solution):
                best_solution = current_set[:]
                self.stats['independent_sets_found'] += 1
                self.stats['max_size_found'] = len(best_solution)
            
            # Pruning based on upper bound
            ub = upper_bound(remaining_vertices, len(current_set))
            if ub <= len(best_solution):
                self.stats['branches_pruned'] += 1
                return
            
            if not remaining_vertices:
                return
            
            # Choose vertex for branching (highest degree first)
            vertex = max(remaining_vertices, key=lambda x: len(graph.get(x, [])))
            remaining_vertices.remove(vertex)
            self.stats['nodes_explored'] += 1
            
            # Branch 1: Include vertex in independent set
            if vertex not in forbidden:
                new_forbidden = forbidden | set(graph.get(vertex, []))
                new_remaining = remaining_vertices - new_forbidden
                branch_and_bound(new_remaining, current_set | {vertex}, new_forbidden)
            
            # Branch 2: Exclude vertex from independent set
            branch_and_bound(remaining_vertices.copy(), current_set, forbidden)
            
            remaining_vertices.add(vertex)  # Restore for other branches
        
        initial_remaining = set(vertices)
        branch_and_bound(initial_remaining, set(), set())
        
        return {
            'independent_set': list(best_solution),
            'size': len(best_solution),
            'is_optimal': True,
            'algorithm': 'branch_and_bound',
            'statistics': self.stats.copy()
        }
    
    def maximum_independent_set_local_search(self, graph: Dict[int, List[int]], iterations: int = 1000) -> Dict:
        """
        Approach 5: Local Search Optimization
        
        Use local search to improve initial solution iteratively.
        
        Time: O(iterations * V^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'independent_set': [], 'size': 0, 'statistics': self.stats}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        
        # Start with greedy solution
        initial_result = self.maximum_independent_set_greedy_degree(graph)
        current_set = set(initial_result['independent_set'])
        best_set = current_set.copy()
        
        def is_independent(vertex_set):
            """Check if vertex set is independent"""
            for u in vertex_set:
                for v in vertex_set:
                    if u != v and v in graph.get(u, []):
                        return False
            return True
        
        def get_neighbors(vertex_set):
            """Get neighboring solutions by adding/removing one vertex"""
            neighbors = []
            
            # Try removing each vertex
            for v in vertex_set:
                new_set = vertex_set - {v}
                neighbors.append(new_set)
            
            # Try adding each non-adjacent vertex
            for v in vertices:
                if v not in vertex_set:
                    # Check if v is adjacent to any vertex in current set
                    can_add = True
                    for u in vertex_set:
                        if v in graph.get(u, []) or u in graph.get(v, []):
                            can_add = False
                            break
                    
                    if can_add:
                        new_set = vertex_set | {v}
                        neighbors.append(new_set)
            
            return neighbors
        
        # Local search iterations
        for iteration in range(iterations):
            self.stats['iterations'] += 1
            
            neighbors = get_neighbors(current_set)
            best_neighbor = None
            best_size = len(current_set)
            
            # Find best neighboring solution
            for neighbor in neighbors:
                self.stats['nodes_explored'] += 1
                if len(neighbor) > best_size and is_independent(neighbor):
                    best_neighbor = neighbor
                    best_size = len(neighbor)
            
            # Move to best neighbor if improvement found
            if best_neighbor is not None:
                current_set = best_neighbor
                if len(current_set) > len(best_set):
                    best_set = current_set.copy()
                    self.stats['independent_sets_found'] += 1
                    self.stats['max_size_found'] = len(best_set)
            else:
                # Local optimum reached
                break
        
        return {
            'independent_set': list(best_set),
            'size': len(best_set),
            'is_local_optimum': True,
            'algorithm': 'local_search',
            'iterations_used': iteration + 1,
            'statistics': self.stats.copy()
        }
    
    def maximum_independent_set_randomized(self, graph: Dict[int, List[int]], trials: int = 100) -> Dict:
        """
        Approach 6: Randomized Algorithm with Multiple Trials
        
        Run randomized greedy algorithm multiple times and return best result.
        
        Time: O(trials * V^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        
        if not graph:
            return {'independent_set': [], 'size': 0, 'statistics': self.stats}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        
        best_set = []
        best_size = 0
        
        for trial in range(trials):
            # Randomized greedy
            remaining = set(vertices)
            current_set = []
            
            # Shuffle vertices for randomization
            shuffled_vertices = vertices[:]
            random.shuffle(shuffled_vertices)
            
            for vertex in shuffled_vertices:
                if vertex in remaining:
                    # Check if vertex can be added (no conflicts)
                    can_add = True
                    for neighbor in graph.get(vertex, []):
                        if neighbor in current_set:
                            can_add = False
                            break
                    
                    if can_add:
                        current_set.append(vertex)
                        remaining.remove(vertex)
                        # Remove neighbors from consideration
                        for neighbor in graph.get(vertex, []):
                            remaining.discard(neighbor)
                    
                    self.stats['nodes_explored'] += 1
            
            # Update best solution
            if len(current_set) > best_size:
                best_set = current_set
                best_size = len(current_set)
                self.stats['independent_sets_found'] += 1
                self.stats['max_size_found'] = best_size
            
            self.stats['iterations'] += 1
        
        return {
            'independent_set': best_set,
            'size': best_size,
            'algorithm': 'randomized_greedy',
            'trials': trials,
            'statistics': self.stats.copy()
        }

def test_independent_set_algorithms():
    """Test all independent set algorithms"""
    solver = IndependentSetAlgorithms()
    
    print("=== Testing Independent Set Algorithms ===")
    
    # Test graphs
    test_graphs = [
        # Path graph P4
        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        
        # Cycle C5
        {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]},
        
        # Complete graph K4
        {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]},
        
        # Star graph
        {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]},
        
        # Bipartite graph
        {0: [3, 4], 1: [3, 4], 2: [3, 4], 3: [0, 1, 2], 4: [0, 1, 2]},
        
        # Empty graph
        {0: [], 1: [], 2: []},
        
        # Triangle
        {0: [1, 2], 1: [0, 2], 2: [0, 1]},
    ]
    
    algorithms = [
        ("Greedy Degree", solver.maximum_independent_set_greedy_degree),
        ("Complement Clique", solver.maximum_independent_set_complement_clique),
        ("Local Search", solver.maximum_independent_set_local_search),
        ("Randomized", solver.maximum_independent_set_randomized),
    ]
    
    for i, graph in enumerate(test_graphs):
        print(f"\n--- Test Graph {i+1} ---")
        print(f"Graph: {graph}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                size = result['size']
                independent_set = result['independent_set']
                stats = result['statistics']
                
                # Validate independence
                valid = validate_independent_set(graph, independent_set)
                status = "✓" if valid else "✗"
                
                print(f"{alg_name:18} | {status} | Size: {size:2} | Set: {sorted(independent_set)} | "
                      f"Explored: {stats.get('nodes_explored', 0):3}")
                
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

def validate_independent_set(graph: Dict[int, List[int]], independent_set: List[int]) -> bool:
    """Validate that the set is indeed independent"""
    for u in independent_set:
        for v in independent_set:
            if u != v and v in graph.get(u, []):
                return False
    return True

def demonstrate_independent_set_concepts():
    """Demonstrate independent set concepts"""
    print("\n=== Independent Set Concepts Demo ===")
    
    graph = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
    
    print(f"Graph: {graph}")
    print(f"Structure: Square with vertices 0-1-2-3-0")
    
    print(f"\nIndependent sets analysis:")
    print(f"• Independent set: vertices with no edges between them")
    print(f"• Maximum independent set: largest such set")
    print(f"• Maximal independent set: cannot add more vertices")
    
    solver = IndependentSetAlgorithms()
    result = solver.maximum_independent_set_greedy_degree(graph)
    
    print(f"\nGreedy algorithm result:")
    print(f"Independent set: {result['independent_set']}")
    print(f"Size: {result['size']}")
    
    print(f"\nValidation:")
    independent_set = result['independent_set']
    print(f"Checking independence...")
    
    for u in independent_set:
        for v in independent_set:
            if u != v:
                adjacent = v in graph.get(u, [])
                print(f"  Vertices {u} and {v}: {'adjacent' if adjacent else 'not adjacent'}")
    
    valid = validate_independent_set(graph, independent_set)
    print(f"Is valid independent set: {valid}")

def demonstrate_algorithm_comparison():
    """Demonstrate comparison between algorithms"""
    print("\n=== Algorithm Comparison Demo ===")
    
    # Create test graph with known maximum independent set
    graph = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4],
        4: [1, 3, 5],
        5: [2, 4]
    }
    
    print(f"Test graph: {graph}")
    print(f"Structure: 2x3 grid graph")
    print(f"Known maximum independent set size: 3")
    
    solver = IndependentSetAlgorithms()
    
    algorithms = [
        ("Greedy Degree", solver.maximum_independent_set_greedy_degree),
        ("Complement Clique", solver.maximum_independent_set_complement_clique),
        ("Local Search", solver.maximum_independent_set_local_search),
        ("Randomized", solver.maximum_independent_set_randomized),
    ]
    
    print(f"\nAlgorithm performance:")
    print(f"{'Algorithm':<18} | {'Size':<4} | {'Set':<12} | {'Explored':<8} | {'Time'}")
    print("-" * 65)
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(graph)
            size = result['size']
            independent_set = sorted(result['independent_set'])
            explored = result['statistics'].get('nodes_explored', 0)
            
            print(f"{alg_name:<18} | {size:<4} | {str(independent_set):<12} | {explored:<8} | Fast")
            
        except Exception as e:
            print(f"{alg_name:<18} | ERROR: {str(e)[:30]}")

def analyze_theoretical_foundations():
    """Analyze theoretical foundations of independent set"""
    print("\n=== Theoretical Foundations ===")
    
    print("Independent Set Theory:")
    
    print("\n1. **Problem Definition:**")
    print("   • Independent set: vertices with no edges between them")
    print("   • Maximum independent set: largest independent set")
    print("   • Independence number α(G): size of maximum independent set")
    
    print("\n2. **Complexity Results:**")
    print("   • Maximum independent set is NP-complete")
    print("   • No polynomial-time approximation better than n^(1-ε)")
    print("   • Greedy gives O(Δ) approximation (Δ = max degree)")
    print("   • Special graph classes have polynomial algorithms")
    
    print("\n3. **Graph Theory Relationships:**")
    print("   • α(G) + τ(G) = n (independence number + vertex cover)")
    print("   • α(G) = ω(Ḡ) (independence in G = clique in complement)")
    print("   • χ(G) ≥ n/α(G) (chromatic number bound)")
    
    print("\n4. **Special Graph Classes:**")
    print("   • Trees: α(G) ≥ n/2 (always large independent set)")
    print("   • Bipartite graphs: α(G) = maximum matching complement")
    print("   • Perfect graphs: α(G) can be found in polynomial time")
    print("   • Planar graphs: α(G) ≥ n/4")
    
    print("\n5. **Algorithm Categories:**")
    print("   • Exact: backtracking, branch-and-bound, dynamic programming")
    print("   • Approximation: greedy, local search, LP rounding")
    print("   • Heuristic: randomized, genetic algorithms, simulated annealing")
    print("   • Special cases: trees, bipartite, cographs")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    print("Independent Set Applications:")
    
    print("\n1. **Scheduling Problems:**")
    print("   • Tasks → Vertices")
    print("   • Conflicts → Edges")
    print("   • Find maximum set of non-conflicting tasks")
    print("   • Maximize parallel execution")
    
    print("\n2. **Facility Location:**")
    print("   • Locations → Vertices")
    print("   • Interference/competition → Edges")
    print("   • Place maximum facilities without conflicts")
    print("   • Cell tower placement, store locations")
    
    print("\n3. **Social Networks:**")
    print("   • People → Vertices")
    print("   • Conflicts/dislikes → Edges")
    print("   • Find largest group with no internal conflicts")
    print("   • Event planning, team formation")
    
    print("\n4. **Resource Allocation:**")
    print("   • Resources → Vertices")
    print("   • Mutual exclusion → Edges")
    print("   • Maximize simultaneous resource usage")
    print("   • Database transactions, CPU scheduling")
    
    print("\n5. **Bioinformatics:**")
    print("   • Genes/proteins → Vertices")
    print("   • Interactions → Edges")
    print("   • Find independent functional modules")
    print("   • Drug target identification")

def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques"""
    print("\n=== Optimization Techniques ===")
    
    print("Independent Set Optimization:")
    
    print("\n1. **Exact Algorithms:**")
    print("   • Dynamic programming on tree decompositions")
    print("   • Integer linear programming formulations")
    print("   • Branch-and-bound with tight upper bounds")
    print("   • Inclusion-exclusion principle")
    
    print("\n2. **Approximation Algorithms:**")
    print("   • Greedy: O(Δ) approximation ratio")
    print("   • LP rounding: O(log n) for some cases")
    print("   • Local search: constant factor for special graphs")
    print("   • PTAS for planar graphs and bounded treewidth")
    
    print("\n3. **Heuristic Improvements:**")
    print("   • Randomized greedy with multiple trials")
    print("   • Simulated annealing and genetic algorithms")
    print("   • Tabu search and variable neighborhood search")
    print("   • Hybrid approaches combining multiple methods")
    
    print("\n4. **Preprocessing Techniques:**")
    print("   • Vertex degree reduction rules")
    print("   • Crown decomposition")
    print("   • Kernelization for parameterized algorithms")
    print("   • Graph simplification and reduction")
    
    print("\n5. **Implementation Optimizations:**")
    print("   • Efficient data structures for neighbor queries")
    print("   • Bit manipulation for set operations")
    print("   • Parallel processing for independent subproblems")
    print("   • Memory-efficient graph representations")

if __name__ == "__main__":
    test_independent_set_algorithms()
    demonstrate_independent_set_concepts()
    demonstrate_algorithm_comparison()
    analyze_theoretical_foundations()
    demonstrate_real_world_applications()
    demonstrate_optimization_techniques()

"""
Independent Set Algorithms and Optimization Concepts:
1. Maximum Independent Set Problem and NP-Completeness
2. Approximation Algorithms and Performance Guarantees
3. Exact Algorithms with Backtracking and Branch-and-Bound
4. Local Search and Metaheuristic Optimization
5. Graph Theory Applications and Real-world Problem Solving

Key Algorithmic Insights:
- Maximum independent set is NP-complete for general graphs
- Greedy algorithms provide reasonable approximations
- Local search can improve solution quality significantly
- Complement graph relationships enable alternative approaches
- Special graph classes admit polynomial-time solutions

Algorithm Strategy:
1. Choose appropriate algorithm based on graph size and structure
2. Use greedy for fast approximate solutions
3. Apply local search for solution improvement
4. Use exact methods for small instances or special cases

Theoretical Foundations:
- Independence number α(G) and its relationships
- Complexity theory and approximation hardness
- Graph theory connections to cliques and vertex covers
- Special graph classes with tractable solutions

Optimization Techniques:
- Sophisticated branching and pruning strategies
- Upper bound estimation for branch-and-bound
- Randomization for escaping local optima
- Preprocessing and graph reduction techniques

Real-world Applications:
- Scheduling and resource allocation problems
- Facility location and interference minimization
- Social network analysis and conflict resolution
- Bioinformatics and computational biology
- Parallel computing and task distribution

This comprehensive implementation provides tools for
solving independent set problems across various domains.
"""
