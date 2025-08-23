"""
Chromatic Number Algorithms - Advanced Implementation
Difficulty: Hard

This file provides comprehensive implementations of chromatic number algorithms,
including exact computation, approximation methods, and specialized techniques for
determining the minimum number of colors needed to properly color a graph.

Key Concepts:
1. Chromatic Number χ(G) Computation
2. Exact Algorithms with Advanced Techniques
3. Approximation Algorithms and Bounds
4. Special Graph Classes and Polynomial Cases
5. Graph Coloring Theory and Applications
6. Advanced Optimization and Heuristics
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import itertools
import random
import heapq
from math import log, ceil

class ChromaticNumberAlgorithms:
    """Advanced chromatic number computation algorithms"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm performance statistics"""
        self.stats = {
            'colorings_tested': 0,
            'branches_explored': 0,
            'bounds_computed': 0,
            'reductions_applied': 0,
            'best_coloring_found': float('inf'),
            'lower_bound': 0,
            'upper_bound': float('inf')
        }
    
    def chromatic_number_exact_exponential(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 1: Exact Exponential Algorithm
        
        Systematically try all possible colorings to find minimum chromatic number.
        
        Time: O(k^n * poly(n)) where k is chromatic number
        Space: O(n)
        """
        self.reset_statistics()
        
        if not graph:
            return {'chromatic_number': 0, 'coloring': {}, 'algorithm': 'exact_exponential'}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        n = len(vertices)
        
        # Calculate initial bounds
        lower_bound = self._calculate_lower_bound(graph, vertices)
        upper_bound = len(vertices)  # Trivial upper bound
        
        self.stats['lower_bound'] = lower_bound
        self.stats['upper_bound'] = upper_bound
        
        def is_valid_coloring(coloring):
            """Check if coloring is valid"""
            for u in graph:
                for v in graph[u]:
                    if u in coloring and v in coloring and coloring[u] == coloring[v]:
                        return False
            return True
        
        def backtrack_coloring(vertex_idx, coloring, num_colors):
            """Backtracking to find valid coloring with num_colors"""
            self.stats['branches_explored'] += 1
            
            if vertex_idx == len(vertices):
                return True  # All vertices colored successfully
            
            vertex = vertices[vertex_idx]
            
            # Try each color
            for color in range(num_colors):
                # Check if color is valid for current vertex
                valid = True
                for neighbor in graph.get(vertex, []):
                    if neighbor in coloring and coloring[neighbor] == color:
                        valid = False
                        break
                
                if valid:
                    coloring[vertex] = color
                    if backtrack_coloring(vertex_idx + 1, coloring, num_colors):
                        return True
                    del coloring[vertex]
            
            return False
        
        # Binary search on number of colors
        best_coloring = None
        
        for k in range(lower_bound, upper_bound + 1):
            self.stats['colorings_tested'] += 1
            coloring = {}
            
            if backtrack_coloring(0, coloring, k):
                best_coloring = coloring.copy()
                self.stats['best_coloring_found'] = k
                break
        
        chromatic_number = self.stats['best_coloring_found']
        if chromatic_number == float('inf'):
            chromatic_number = upper_bound
            best_coloring = {v: i for i, v in enumerate(vertices)}
        
        return {
            'chromatic_number': chromatic_number,
            'coloring': best_coloring,
            'is_optimal': True,
            'algorithm': 'exact_exponential',
            'bounds': (lower_bound, upper_bound),
            'statistics': self.stats.copy()
        }
    
    def chromatic_number_inclusion_exclusion(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 2: Inclusion-Exclusion Principle
        
        Use inclusion-exclusion to count proper colorings and find chromatic number.
        
        Time: O(2^n * poly(n))
        Space: O(2^n)
        """
        self.reset_statistics()
        
        if not graph:
            return {'chromatic_number': 0, 'algorithm': 'inclusion_exclusion'}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        n = len(vertices)
        
        # Build edge list
        edges = []
        for u in graph:
            for v in graph[u]:
                if u < v:
                    edges.append((u, v))
        
        def chromatic_polynomial(k):
            """Compute chromatic polynomial P(G, k) using inclusion-exclusion"""
            result = 0
            
            # Iterate over all subsets of edges
            for edge_subset_bits in range(1 << len(edges)):
                self.stats['branches_explored'] += 1
                
                edge_subset = []
                for i in range(len(edges)):
                    if edge_subset_bits & (1 << i):
                        edge_subset.append(edges[i])
                
                # Count connected components when edge_subset is contracted
                components = self._count_components_after_contraction(vertices, edge_subset)
                
                # Inclusion-exclusion: alternate signs
                sign = (-1) ** len(edge_subset)
                result += sign * (k ** components)
            
            return result
        
        # Find chromatic number by finding smallest k where P(G, k) > 0
        lower_bound = self._calculate_lower_bound(graph, vertices)
        
        for k in range(lower_bound, n + 1):
            self.stats['colorings_tested'] += 1
            if chromatic_polynomial(k) > 0:
                return {
                    'chromatic_number': k,
                    'algorithm': 'inclusion_exclusion',
                    'chromatic_polynomial_value': chromatic_polynomial(k),
                    'statistics': self.stats.copy()
                }
        
        return {
            'chromatic_number': n,
            'algorithm': 'inclusion_exclusion',
            'statistics': self.stats.copy()
        }
    
    def chromatic_number_dsatur_based(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 3: DSATUR-based Upper Bound with Improvements
        
        Use DSATUR heuristic and try to improve the bound iteratively.
        
        Time: O(n^3)
        Space: O(n^2)
        """
        self.reset_statistics()
        
        if not graph:
            return {'chromatic_number': 0, 'coloring': {}, 'algorithm': 'dsatur_based'}
        
        vertices = set(v for v in graph.keys()) | set(v for neighbors in graph.values() for v in neighbors)
        
        def dsatur_coloring():
            """DSATUR algorithm implementation"""
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
                self.stats['branches_explored'] += 1
            
            return coloring
        
        # Run DSATUR multiple times with different vertex orderings
        best_coloring = None
        best_colors = float('inf')
        
        for trial in range(10):  # Multiple trials
            self.stats['colorings_tested'] += 1
            
            # Randomize the tie-breaking in DSATUR
            if trial > 0:
                vertices_list = list(vertices)
                random.shuffle(vertices_list)
            
            coloring = dsatur_coloring()
            colors_used = len(set(coloring.values()))
            
            if colors_used < best_colors:
                best_colors = colors_used
                best_coloring = coloring
                self.stats['best_coloring_found'] = colors_used
        
        # Try to improve with local search
        improved_coloring = self._local_search_improvement(graph, best_coloring)
        improved_colors = len(set(improved_coloring.values()))
        
        if improved_colors < best_colors:
            best_colors = improved_colors
            best_coloring = improved_coloring
        
        lower_bound = self._calculate_lower_bound(graph, list(vertices))
        
        return {
            'chromatic_number': best_colors,
            'coloring': best_coloring,
            'is_upper_bound': True,
            'lower_bound': lower_bound,
            'algorithm': 'dsatur_based',
            'statistics': self.stats.copy()
        }
    
    def chromatic_number_branch_and_bound(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 4: Advanced Branch and Bound
        
        Sophisticated branch and bound with multiple pruning techniques.
        
        Time: O(2^n) with aggressive pruning
        Space: O(n)
        """
        self.reset_statistics()
        
        if not graph:
            return {'chromatic_number': 0, 'coloring': {}, 'algorithm': 'branch_and_bound'}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        
        # Calculate bounds
        lower_bound = self._calculate_lower_bound(graph, vertices)
        upper_bound = self._calculate_upper_bound(graph, vertices)
        
        self.stats['lower_bound'] = lower_bound
        self.stats['upper_bound'] = upper_bound
        
        best_coloring = None
        best_colors = upper_bound
        
        def advanced_lower_bound(partial_coloring, remaining_vertices):
            """Advanced lower bound computation"""
            # Current colors used
            current_colors = len(set(partial_coloring.values())) if partial_coloring else 0
            
            # Build subgraph of remaining vertices
            subgraph_edges = 0
            for u in remaining_vertices:
                for v in graph.get(u, []):
                    if v in remaining_vertices and u < v:
                        subgraph_edges += 1
            
            # Clique-based lower bound for remaining subgraph
            if remaining_vertices:
                max_clique_size = self._estimate_max_clique(graph, remaining_vertices)
                remaining_lower_bound = max_clique_size
            else:
                remaining_lower_bound = 0
            
            return max(current_colors + remaining_lower_bound, current_colors, lower_bound)
        
        def branch_and_bound(vertex_idx, coloring, remaining_vertices):
            """Branch and bound search"""
            nonlocal best_coloring, best_colors
            
            self.stats['branches_explored'] += 1
            
            if vertex_idx == len(vertices):
                colors_used = len(set(coloring.values()))
                if colors_used < best_colors:
                    best_colors = colors_used
                    best_coloring = coloring.copy()
                    self.stats['best_coloring_found'] = colors_used
                return
            
            # Pruning: check lower bound
            lb = advanced_lower_bound(coloring, remaining_vertices)
            if lb >= best_colors:
                return
            
            vertex = vertices[vertex_idx]
            new_remaining = remaining_vertices - {vertex}
            
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in graph.get(vertex, []):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            # Try existing colors first
            existing_colors = set(coloring.values()) if coloring else set()
            
            for color in sorted(existing_colors):
                if color not in neighbor_colors:
                    coloring[vertex] = color
                    branch_and_bound(vertex_idx + 1, coloring, new_remaining)
                    del coloring[vertex]
            
            # Try a new color if it doesn't exceed bound
            new_color = len(existing_colors)
            if new_color < best_colors - 1:  # -1 because we're adding one more color
                coloring[vertex] = new_color
                branch_and_bound(vertex_idx + 1, coloring, new_remaining)
                del coloring[vertex]
        
        # Start branch and bound
        branch_and_bound(0, {}, set(vertices))
        
        return {
            'chromatic_number': best_colors,
            'coloring': best_coloring,
            'is_optimal': True,
            'algorithm': 'branch_and_bound',
            'bounds': (lower_bound, upper_bound),
            'statistics': self.stats.copy()
        }
    
    def chromatic_number_special_cases(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 5: Special Case Recognition and Algorithms
        
        Detect special graph classes and apply optimal algorithms.
        
        Time: Varies by graph class
        Space: O(n + m)
        """
        self.reset_statistics()
        
        if not graph:
            return {'chromatic_number': 0, 'algorithm': 'special_cases'}
        
        vertices = set(v for v in graph.keys()) | set(v for neighbors in graph.values() for v in neighbors)
        
        # Check for bipartite graph
        bipartite_result = self._check_bipartite(graph, vertices)
        if bipartite_result['is_bipartite']:
            if bipartite_result['has_edges']:
                return {
                    'chromatic_number': 2,
                    'coloring': bipartite_result['coloring'],
                    'graph_class': 'bipartite',
                    'algorithm': 'special_cases',
                    'statistics': self.stats.copy()
                }
            else:
                return {
                    'chromatic_number': 1,
                    'coloring': {v: 0 for v in vertices},
                    'graph_class': 'independent_set',
                    'algorithm': 'special_cases',
                    'statistics': self.stats.copy()
                }
        
        # Check for complete graph
        if self._is_complete_graph(graph, vertices):
            return {
                'chromatic_number': len(vertices),
                'coloring': {v: i for i, v in enumerate(vertices)},
                'graph_class': 'complete',
                'algorithm': 'special_cases',
                'statistics': self.stats.copy()
            }
        
        # Check for tree
        tree_result = self._check_tree(graph, vertices)
        if tree_result['is_tree']:
            if len(vertices) > 1:
                return {
                    'chromatic_number': 2,
                    'coloring': tree_result['coloring'],
                    'graph_class': 'tree',
                    'algorithm': 'special_cases',
                    'statistics': self.stats.copy()
                }
            else:
                return {
                    'chromatic_number': 1,
                    'coloring': {list(vertices)[0]: 0},
                    'graph_class': 'single_vertex',
                    'algorithm': 'special_cases',
                    'statistics': self.stats.copy()
                }
        
        # Check for cycle
        cycle_result = self._check_cycle(graph, vertices)
        if cycle_result['is_cycle']:
            cycle_length = len(vertices)
            chromatic_number = 3 if cycle_length % 2 == 1 else 2
            return {
                'chromatic_number': chromatic_number,
                'coloring': cycle_result['coloring'],
                'graph_class': f'cycle_C{cycle_length}',
                'algorithm': 'special_cases',
                'statistics': self.stats.copy()
            }
        
        # If no special case detected, fall back to heuristic
        return {
            'chromatic_number': None,
            'graph_class': 'general',
            'algorithm': 'special_cases',
            'message': 'No special case detected',
            'statistics': self.stats.copy()
        }
    
    def chromatic_number_approximation(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 6: Approximation Algorithm with Performance Guarantees
        
        Multiple approximation algorithms with theoretical guarantees.
        
        Time: O(n^2 log n)
        Space: O(n^2)
        """
        self.reset_statistics()
        
        if not graph:
            return {'chromatic_number': 0, 'algorithm': 'approximation'}
        
        vertices = list(set(v for v in graph.keys()) | 
                       set(v for neighbors in graph.values() for v in neighbors))
        
        # Algorithm 1: Greedy with degree ordering
        def greedy_degree_approximation():
            """Greedy algorithm with degree-based ordering"""
            # Sort vertices by degree (descending)
            degree = {v: len(graph.get(v, [])) for v in vertices}
            sorted_vertices = sorted(vertices, key=lambda x: degree[x], reverse=True)
            
            coloring = {}
            for vertex in sorted_vertices:
                neighbor_colors = set()
                for neighbor in graph.get(vertex, []):
                    if neighbor in coloring:
                        neighbor_colors.add(coloring[neighbor])
                
                color = 0
                while color in neighbor_colors:
                    color += 1
                
                coloring[vertex] = color
                self.stats['branches_explored'] += 1
            
            return coloring
        
        # Algorithm 2: Smallest-last ordering
        def smallest_last_approximation():
            """Smallest-last ordering heuristic"""
            vertices_copy = vertices[:]
            ordering = []
            
            # Build temporary graph
            temp_graph = defaultdict(set)
            for u in graph:
                for v in graph[u]:
                    temp_graph[u].add(v)
                    temp_graph[v].add(u)
            
            remaining = set(vertices)
            
            while remaining:
                # Find vertex with minimum degree
                min_degree = float('inf')
                min_vertex = None
                
                for v in remaining:
                    degree = len(temp_graph[v] & remaining)
                    if degree < min_degree:
                        min_degree = degree
                        min_vertex = v
                
                ordering.append(min_vertex)
                remaining.remove(min_vertex)
                
                # Update graph
                for neighbor in temp_graph[min_vertex]:
                    temp_graph[neighbor].discard(min_vertex)
            
            # Color in reverse order
            coloring = {}
            for vertex in reversed(ordering):
                neighbor_colors = set()
                for neighbor in graph.get(vertex, []):
                    if neighbor in coloring:
                        neighbor_colors.add(coloring[neighbor])
                
                color = 0
                while color in neighbor_colors:
                    color += 1
                
                coloring[vertex] = color
                self.stats['branches_explored'] += 1
            
            return coloring
        
        # Algorithm 3: Random ordering (multiple trials)
        def random_approximation():
            """Random ordering with multiple trials"""
            best_coloring = None
            best_colors = float('inf')
            
            for trial in range(20):
                random_vertices = vertices[:]
                random.shuffle(random_vertices)
                
                coloring = {}
                for vertex in random_vertices:
                    neighbor_colors = set()
                    for neighbor in graph.get(vertex, []):
                        if neighbor in coloring:
                            neighbor_colors.add(coloring[neighbor])
                    
                    color = 0
                    while color in neighbor_colors:
                        color += 1
                    
                    coloring[vertex] = color
                
                colors_used = len(set(coloring.values()))
                if colors_used < best_colors:
                    best_colors = colors_used
                    best_coloring = coloring
                
                self.stats['branches_explored'] += 1
            
            return best_coloring
        
        # Run all approximation algorithms
        algorithms = [
            ("greedy_degree", greedy_degree_approximation),
            ("smallest_last", smallest_last_approximation),
            ("random_multiple", random_approximation),
        ]
        
        best_result = None
        best_colors = float('inf')
        
        for alg_name, alg_func in algorithms:
            self.stats['colorings_tested'] += 1
            coloring = alg_func()
            colors_used = len(set(coloring.values()))
            
            if colors_used < best_colors:
                best_colors = colors_used
                best_result = {
                    'chromatic_number': colors_used,
                    'coloring': coloring,
                    'best_algorithm': alg_name
                }
        
        # Calculate theoretical bounds
        max_degree = max(len(graph.get(v, [])) for v in vertices) if vertices else 0
        lower_bound = self._calculate_lower_bound(graph, vertices)
        
        return {
            **best_result,
            'is_approximation': True,
            'algorithm': 'approximation',
            'approximation_ratio': f'≤ {max_degree + 1}',
            'lower_bound': lower_bound,
            'upper_bound_guarantee': max_degree + 1,
            'statistics': self.stats.copy()
        }
    
    def _calculate_lower_bound(self, graph: Dict[int, List[int]], vertices: List[int]) -> int:
        """Calculate lower bound on chromatic number"""
        self.stats['bounds_computed'] += 1
        
        if not vertices:
            return 0
        
        # Lower bound 1: Maximum clique size
        max_clique_size = self._estimate_max_clique(graph, vertices)
        
        # Lower bound 2: |V| / α(G) where α(G) is independence number
        # Use greedy approximation for independence number
        independence_approx = self._estimate_independence_number(graph, vertices)
        independence_bound = len(vertices) // independence_approx if independence_approx > 0 else len(vertices)
        
        return max(max_clique_size, independence_bound, 1)
    
    def _calculate_upper_bound(self, graph: Dict[int, List[int]], vertices: List[int]) -> int:
        """Calculate upper bound on chromatic number"""
        if not vertices:
            return 0
        
        # Brooks' theorem: χ(G) ≤ Δ(G) unless G is complete or odd cycle
        max_degree = max(len(graph.get(v, [])) for v in vertices) if vertices else 0
        
        return min(len(vertices), max_degree + 1)
    
    def _estimate_max_clique(self, graph: Dict[int, List[int]], vertices: List[int]) -> int:
        """Estimate maximum clique size (greedy approximation)"""
        max_clique_size = 1
        
        for start_vertex in vertices[:min(10, len(vertices))]:  # Try few starting points
            clique = [start_vertex]
            candidates = set(graph.get(start_vertex, []))
            
            while candidates:
                # Choose vertex with maximum connections to current clique
                best_vertex = None
                max_connections = -1
                
                for v in candidates:
                    connections = sum(1 for u in clique if u in graph.get(v, []))
                    if connections > max_connections:
                        max_connections = connections
                        best_vertex = v
                
                if best_vertex and max_connections == len(clique):
                    clique.append(best_vertex)
                    # Update candidates
                    new_candidates = set()
                    for v in candidates:
                        if v != best_vertex and all(v in graph.get(u, []) for u in clique):
                            new_candidates.add(v)
                    candidates = new_candidates
                else:
                    break
            
            max_clique_size = max(max_clique_size, len(clique))
        
        return max_clique_size
    
    def _estimate_independence_number(self, graph: Dict[int, List[int]], vertices: List[int]) -> int:
        """Estimate independence number (greedy approximation)"""
        independent_set = []
        remaining = set(vertices)
        
        while remaining:
            # Choose vertex with minimum degree in remaining graph
            min_degree = float('inf')
            min_vertex = None
            
            for v in remaining:
                degree = len([n for n in graph.get(v, []) if n in remaining])
                if degree < min_degree:
                    min_degree = degree
                    min_vertex = v
            
            if min_vertex:
                independent_set.append(min_vertex)
                remaining.remove(min_vertex)
                # Remove neighbors
                for neighbor in graph.get(min_vertex, []):
                    remaining.discard(neighbor)
        
        return len(independent_set)
    
    def _local_search_improvement(self, graph: Dict[int, List[int]], coloring: Dict[int, int]) -> Dict[int, int]:
        """Apply local search to improve coloring"""
        improved_coloring = coloring.copy()
        colors_used = set(coloring.values())
        
        # Try to remove the highest color
        max_color = max(colors_used) if colors_used else 0
        vertices_with_max_color = [v for v in coloring if coloring[v] == max_color]
        
        # Try to recolor vertices with maximum color
        for vertex in vertices_with_max_color:
            neighbor_colors = set()
            for neighbor in graph.get(vertex, []):
                if neighbor in improved_coloring and neighbor != vertex:
                    neighbor_colors.add(improved_coloring[neighbor])
            
            # Find available color < max_color
            for color in range(max_color):
                if color not in neighbor_colors:
                    improved_coloring[vertex] = color
                    break
        
        return improved_coloring
    
    def _count_components_after_contraction(self, vertices: List[int], edge_subset: List[Tuple[int, int]]) -> int:
        """Count connected components after contracting edge subset"""
        # Union-Find to track connected components
        parent = {v: v for v in vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Contract edges in subset
        for u, v in edge_subset:
            union(u, v)
        
        # Count unique components
        components = len(set(find(v) for v in vertices))
        return components
    
    def _check_bipartite(self, graph: Dict[int, List[int]], vertices: Set[int]) -> Dict:
        """Check if graph is bipartite"""
        color = {}
        
        def bfs_color(start):
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                for neighbor in graph.get(node, []):
                    if neighbor not in color:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
            return True
        
        # Check all components
        for v in vertices:
            if v not in color:
                if not bfs_color(v):
                    return {'is_bipartite': False}
        
        # Check if there are any edges
        has_edges = any(graph.get(v, []) for v in vertices)
        
        return {
            'is_bipartite': True,
            'has_edges': has_edges,
            'coloring': color
        }
    
    def _is_complete_graph(self, graph: Dict[int, List[int]], vertices: Set[int]) -> bool:
        """Check if graph is complete"""
        n = len(vertices)
        for v in vertices:
            if len(graph.get(v, [])) != n - 1:
                return False
        return True
    
    def _check_tree(self, graph: Dict[int, List[int]], vertices: Set[int]) -> Dict:
        """Check if graph is a tree and provide 2-coloring if so"""
        n = len(vertices)
        if n == 0:
            return {'is_tree': True, 'coloring': {}}
        
        # Count edges
        edge_count = sum(len(graph.get(v, [])) for v in vertices) // 2
        
        if edge_count != n - 1:
            return {'is_tree': False}
        
        # Check connectivity using BFS
        start = next(iter(vertices))
        visited = set()
        queue = deque([start])
        color = {start: 0}
        
        while queue:
            node = queue.popleft()
            visited.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
        
        is_tree = len(visited) == n
        return {
            'is_tree': is_tree,
            'coloring': color if is_tree else {}
        }
    
    def _check_cycle(self, graph: Dict[int, List[int]], vertices: Set[int]) -> Dict:
        """Check if graph is a cycle"""
        n = len(vertices)
        if n < 3:
            return {'is_cycle': False}
        
        # Check if every vertex has degree 2
        for v in vertices:
            if len(graph.get(v, [])) != 2:
                return {'is_cycle': False}
        
        # Check if it forms a single cycle
        start = next(iter(vertices))
        current = start
        prev = None
        path = [current]
        
        for i in range(n):
            neighbors = [n for n in graph.get(current, []) if n != prev]
            if len(neighbors) != 1:
                return {'is_cycle': False}
            
            prev = current
            current = neighbors[0]
            
            if i < n - 1:
                path.append(current)
            elif current != start:
                return {'is_cycle': False}
        
        # Create coloring for cycle
        coloring = {}
        for i, v in enumerate(path):
            if n % 2 == 1:  # Odd cycle needs 3 colors
                coloring[v] = i % 3
            else:  # Even cycle needs 2 colors
                coloring[v] = i % 2
        
        return {
            'is_cycle': True,
            'coloring': coloring
        }

def test_chromatic_number_algorithms():
    """Test all chromatic number algorithms"""
    solver = ChromaticNumberAlgorithms()
    
    print("=== Testing Chromatic Number Algorithms ===")
    
    # Test graphs with known chromatic numbers
    test_graphs = [
        # (graph, name, expected_chromatic_number)
        ({}, "Empty", 0),
        ({0: []}, "Single vertex", 1),
        ({0: [], 1: [], 2: []}, "Independent set", 1),
        ({0: [1], 1: [0]}, "Edge", 2),
        ({0: [1, 2], 1: [0, 2], 2: [0, 1]}, "Triangle", 3),
        ({0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}, "4-cycle", 2),
        ({0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}, "Star", 2),
        ({0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]}, "5-cycle", 3),
        ({0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}, "K4", 4),
    ]
    
    algorithms = [
        ("Special Cases", solver.chromatic_number_special_cases),
        ("DSATUR-based", solver.chromatic_number_dsatur_based),
        ("Approximation", solver.chromatic_number_approximation),
    ]
    
    for graph, name, expected in test_graphs:
        print(f"\n--- {name} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                
                if result.get('chromatic_number') is None:
                    print(f"{alg_name:15} | SKIP | {result.get('message', 'N/A')}")
                    continue
                
                chi = result['chromatic_number']
                correct = "✓" if chi == expected else "✗"
                
                graph_class = result.get('graph_class', 'general')
                
                print(f"{alg_name:15} | {correct} | χ = {chi} | Class: {graph_class}")
                
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

def demonstrate_chromatic_number_theory():
    """Demonstrate chromatic number theory"""
    print("\n=== Chromatic Number Theory ===")
    
    print("Chromatic Number χ(G) Theory:")
    
    print("\n1. **Fundamental Bounds:**")
    print("   • ω(G) ≤ χ(G) ≤ Δ(G) + 1")
    print("   • ω(G) = clique number, Δ(G) = maximum degree")
    print("   • Brooks' theorem: χ(G) ≤ Δ(G) unless complete or odd cycle")
    print("   • χ(G) ≥ |V|/α(G) where α(G) = independence number")
    
    print("\n2. **Special Graph Classes:**")
    print("   • Bipartite graphs: χ(G) = 2 (if connected with edges)")
    print("   • Trees: χ(G) = 2 (if more than one vertex)")
    print("   • Odd cycles: χ(G) = 3")
    print("   • Even cycles: χ(G) = 2")
    print("   • Complete graphs: χ(G) = |V|")
    print("   • Planar graphs: χ(G) ≤ 4 (Four Color Theorem)")
    
    print("\n3. **Complexity Results:**")
    print("   • Computing χ(G) is NP-complete")
    print("   • k-coloring is NP-complete for k ≥ 3")
    print("   • 2-coloring (bipartiteness) is polynomial")
    print("   • No good approximation algorithms exist")
    
    print("\n4. **Exact Algorithms:**")
    print("   • Exponential algorithms with various optimizations")
    print("   • Inclusion-exclusion principle")
    print("   • Branch-and-bound with sophisticated pruning")
    print("   • Specialized algorithms for graph classes")
    
    print("\n5. **Practical Approaches:**")
    print("   • DSATUR and other construction heuristics")
    print("   • Local search and improvement")
    print("   • Hybrid exact/heuristic methods")
    print("   • Special case recognition")

if __name__ == "__main__":
    test_chromatic_number_algorithms()
    demonstrate_chromatic_number_theory()

"""
Chromatic Number and Advanced Graph Coloring Theory:
1. Exact Algorithms for Chromatic Number Computation
2. Special Graph Class Recognition and Optimal Algorithms
3. Approximation Methods and Theoretical Bounds
4. Advanced Branch-and-Bound with Sophisticated Pruning
5. Graph Theory Applications and Complexity Analysis

Key Algorithmic Insights:
- Chromatic number computation is NP-complete for general graphs
- Special graph classes admit polynomial-time algorithms
- Multiple lower and upper bound techniques improve exact algorithms
- Heuristic methods provide practical solutions for large instances
- Graph structure recognition enables optimal specialized algorithms

Algorithm Strategy:
1. Recognize special graph structure (bipartite, tree, cycle, complete)
2. Apply optimal specialized algorithm if possible
3. Use exact exponential algorithms for small general graphs
4. Apply advanced heuristics for larger instances
5. Combine multiple approaches for best results

Theoretical Foundations:
- Chromatic number bounds and relationships
- Brooks' theorem and structural results
- Four Color Theorem for planar graphs
- Complexity theory and hardness results
- Perfect graph theory and special cases

Advanced Techniques:
- Sophisticated branching and pruning in exponential algorithms
- Inclusion-exclusion principle for exact computation
- Multiple approximation and heuristic approaches
- Local search and iterative improvement methods
- Graph reduction and preprocessing techniques

Real-world Applications:
- Course and exam scheduling optimization
- Frequency assignment in wireless networks
- Register allocation in compiler optimization
- Map coloring and visualization
- Conflict resolution in resource allocation

This comprehensive implementation provides state-of-the-art
algorithms for the fundamental chromatic number problem.
"""
