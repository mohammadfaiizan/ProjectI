"""
Graph Isomorphism Advanced - Sophisticated Algorithms
Difficulty: Hard

Advanced graph isomorphism algorithms including canonical labeling,
automorphism group computation, and practical isomorphism testing.
This extends beyond basic isomorphism to cover advanced techniques.

Key Concepts:
1. Canonical Labeling Algorithms
2. Weisfeiler-Lehman Algorithm
3. Nauty Algorithm (simplified)
4. Automorphism Group Computation
5. Graph Invariants and Certificates
6. Practical Isomorphism Testing
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, Counter
import itertools
import hashlib

class GraphIsomorphismAdvanced:
    """Advanced graph isomorphism algorithms"""
    
    def __init__(self):
        self.graph1 = defaultdict(set)
        self.graph2 = defaultdict(set)
        self.vertices1 = set()
        self.vertices2 = set()
    
    def set_graphs(self, edges1: List[Tuple[int, int]], edges2: List[Tuple[int, int]]):
        """Set the two graphs to compare"""
        self.graph1.clear()
        self.graph2.clear()
        self.vertices1.clear()
        self.vertices2.clear()
        
        for u, v in edges1:
            self.graph1[u].add(v)
            self.graph1[v].add(u)
            self.vertices1.add(u)
            self.vertices1.add(v)
        
        for u, v in edges2:
            self.graph2[u].add(v)
            self.graph2[v].add(u)
            self.vertices2.add(u)
            self.vertices2.add(v)
    
    def are_isomorphic_weisfeiler_lehman(self, iterations: int = 10) -> bool:
        """
        Approach 1: Weisfeiler-Lehman Algorithm
        
        Iterative refinement of vertex labels based on neighborhood structure.
        
        Time: O(k * (V + E)), Space: O(V)
        """
        if len(self.vertices1) != len(self.vertices2):
            return False
        
        # Initialize labels (degree-based)
        labels1 = {v: len(self.graph1[v]) for v in self.vertices1}
        labels2 = {v: len(self.graph2[v]) for v in self.vertices2}
        
        for iteration in range(iterations):
            # Refine labels based on neighbor labels
            new_labels1 = {}
            new_labels2 = {}
            
            for v in self.vertices1:
                neighbor_labels = sorted([labels1[u] for u in self.graph1[v]])
                new_labels1[v] = hash((labels1[v], tuple(neighbor_labels)))
            
            for v in self.vertices2:
                neighbor_labels = sorted([labels2[u] for u in self.graph2[v]])
                new_labels2[v] = hash((labels2[v], tuple(neighbor_labels)))
            
            # Check if label multisets are the same
            multiset1 = Counter(new_labels1.values())
            multiset2 = Counter(new_labels2.values())
            
            if multiset1 != multiset2:
                return False
            
            # Check for convergence
            if new_labels1 == labels1 and new_labels2 == labels2:
                break
            
            labels1, labels2 = new_labels1, new_labels2
        
        return True
    
    def get_canonical_labeling_nauty_style(self, graph: Dict[int, Set[int]], 
                                         vertices: Set[int]) -> Tuple[str, Dict[int, int]]:
        """
        Approach 2: Nauty-style Canonical Labeling
        
        Simplified version of nauty algorithm for canonical labeling.
        
        Time: Exponential worst case, polynomial average case
        Space: O(V!)
        """
        n = len(vertices)
        vertex_list = sorted(vertices)
        
        # Create initial partition based on degrees
        degree_partition = defaultdict(list)
        for v in vertices:
            degree = len(graph[v])
            degree_partition[degree].append(v)
        
        # Refine partition using automorphism-preserving refinement
        partition = list(degree_partition.values())
        partition = self._refine_partition(graph, partition)
        
        # Find canonical labeling
        best_labeling = None
        best_canonical = None
        
        # Try all possible labelings (simplified - in practice, use backtracking)
        for perm in itertools.permutations(vertex_list):
            labeling = {vertex_list[i]: i for i in range(n)}
            canonical = self._compute_canonical_form(graph, vertices, labeling)
            
            if best_canonical is None or canonical < best_canonical:
                best_canonical = canonical
                best_labeling = labeling
        
        return best_canonical, best_labeling
    
    def are_isomorphic_canonical_labeling(self) -> bool:
        """
        Approach 3: Canonical Labeling Comparison
        
        Compare canonical forms of both graphs.
        
        Time: Exponential worst case, Space: O(V!)
        """
        if len(self.vertices1) != len(self.vertices2):
            return False
        
        canonical1, _ = self.get_canonical_labeling_nauty_style(self.graph1, self.vertices1)
        canonical2, _ = self.get_canonical_labeling_nauty_style(self.graph2, self.vertices2)
        
        return canonical1 == canonical2
    
    def compute_automorphism_group_size(self, graph: Dict[int, Set[int]], 
                                      vertices: Set[int]) -> int:
        """
        Approach 4: Automorphism Group Size Computation
        
        Count the number of automorphisms (isomorphisms to itself).
        
        Time: Exponential, Space: O(V!)
        """
        n = len(vertices)
        if n == 0:
            return 1
        
        vertex_list = sorted(vertices)
        automorphism_count = 0
        
        # Check all permutations
        for perm in itertools.permutations(vertex_list):
            mapping = {vertex_list[i]: perm[i] for i in range(n)}
            
            if self._is_valid_mapping(graph, vertices, mapping):
                automorphism_count += 1
        
        return automorphism_count
    
    def find_graph_invariants(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> Dict:
        """
        Approach 5: Graph Invariants Computation
        
        Compute various graph invariants for isomorphism testing.
        
        Time: O(V^3), Space: O(V^2)
        """
        n = len(vertices)
        
        invariants = {
            'vertex_count': n,
            'edge_count': sum(len(adj) for adj in graph.values()) // 2,
            'degree_sequence': sorted([len(graph[v]) for v in vertices]),
            'triangle_count': self._count_triangles(graph, vertices),
            'diameter': self._compute_diameter(graph, vertices),
            'girth': self._compute_girth(graph, vertices),
            'chromatic_number_upper_bound': self._chromatic_number_upper_bound(graph, vertices),
        }
        
        # Spectral invariants (simplified)
        invariants['eigenvalue_sum'] = sum(invariants['degree_sequence'])
        
        # Distance-based invariants
        distance_matrix = self._compute_distance_matrix(graph, vertices)
        invariants['distance_sum'] = sum(sum(row) for row in distance_matrix.values())
        
        return invariants
    
    def are_isomorphic_invariant_based(self) -> bool:
        """
        Approach 6: Invariant-based Isomorphism Test
        
        Quick test using graph invariants.
        
        Time: O(V^3), Space: O(V^2)
        """
        invariants1 = self.find_graph_invariants(self.graph1, self.vertices1)
        invariants2 = self.find_graph_invariants(self.graph2, self.vertices2)
        
        # Compare all invariants
        for key in invariants1:
            if invariants1[key] != invariants2[key]:
                return False
        
        return True
    
    def find_isomorphism_mapping_advanced(self) -> Optional[Dict[int, int]]:
        """
        Approach 7: Advanced Isomorphism Mapping
        
        Find actual vertex mapping using sophisticated techniques.
        
        Time: Exponential with pruning, Space: O(V)
        """
        if len(self.vertices1) != len(self.vertices2):
            return None
        
        # Quick invariant check
        if not self.are_isomorphic_invariant_based():
            return None
        
        # Use degree-based initial partitioning
        degree_classes1 = self._partition_by_degree(self.graph1, self.vertices1)
        degree_classes2 = self._partition_by_degree(self.graph2, self.vertices2)
        
        if [len(cls) for cls in degree_classes1] != [len(cls) for cls in degree_classes2]:
            return None
        
        # Backtracking search with pruning
        mapping = {}
        return self._backtrack_mapping(degree_classes1, degree_classes2, mapping, 0)
    
    def _refine_partition(self, graph: Dict[int, Set[int]], partition: List[List[int]]) -> List[List[int]]:
        """Refine partition using neighborhood information"""
        changed = True
        
        while changed:
            changed = False
            new_partition = []
            
            for cell in partition:
                if len(cell) <= 1:
                    new_partition.append(cell)
                    continue
                
                # Group vertices by their neighborhood signature
                signatures = {}
                for v in cell:
                    # Count neighbors in each partition cell
                    signature = []
                    for other_cell in partition:
                        count = sum(1 for u in other_cell if u in graph[v])
                        signature.append(count)
                    
                    signature = tuple(signature)
                    if signature not in signatures:
                        signatures[signature] = []
                    signatures[signature].append(v)
                
                if len(signatures) > 1:
                    changed = True
                
                new_partition.extend(signatures.values())
            
            partition = new_partition
        
        return partition
    
    def _compute_canonical_form(self, graph: Dict[int, Set[int]], vertices: Set[int], 
                               labeling: Dict[int, int]) -> str:
        """Compute canonical adjacency matrix string"""
        n = len(vertices)
        matrix = [[0] * n for _ in range(n)]
        
        for u in vertices:
            for v in graph[u]:
                i, j = labeling[u], labeling[v]
                matrix[i][j] = 1
        
        # Convert to string
        return ''.join(''.join(map(str, row)) for row in matrix)
    
    def _is_valid_mapping(self, graph: Dict[int, Set[int]], vertices: Set[int], 
                         mapping: Dict[int, int]) -> bool:
        """Check if mapping preserves adjacency"""
        for u in vertices:
            for v in graph[u]:
                if mapping[v] not in graph[mapping[u]]:
                    return False
        return True
    
    def _count_triangles(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> int:
        """Count triangles in graph"""
        count = 0
        vertex_list = list(vertices)
        
        for i in range(len(vertex_list)):
            for j in range(i + 1, len(vertex_list)):
                for k in range(j + 1, len(vertex_list)):
                    u, v, w = vertex_list[i], vertex_list[j], vertex_list[k]
                    if v in graph[u] and w in graph[u] and w in graph[v]:
                        count += 1
        
        return count
    
    def _compute_diameter(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> int:
        """Compute graph diameter"""
        if not vertices:
            return 0
        
        max_distance = 0
        
        for start in vertices:
            distances = self._bfs_distances(graph, start)
            for v in vertices:
                if v in distances:
                    max_distance = max(max_distance, distances[v])
                else:
                    return float('inf')  # Disconnected
        
        return max_distance
    
    def _compute_girth(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> int:
        """Compute graph girth (shortest cycle length)"""
        min_cycle = float('inf')
        
        for start in vertices:
            # BFS to find shortest cycle containing start
            queue = [(start, -1, 0)]  # (vertex, parent, distance)
            visited = {start: 0}
            
            while queue:
                v, parent, dist = queue.pop(0)
                
                for u in graph[v]:
                    if u == parent:
                        continue
                    
                    if u in visited:
                        cycle_length = dist + visited[u] + 1
                        min_cycle = min(min_cycle, cycle_length)
                    else:
                        visited[u] = dist + 1
                        queue.append((u, v, dist + 1))
        
        return min_cycle if min_cycle != float('inf') else 0
    
    def _chromatic_number_upper_bound(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> int:
        """Upper bound on chromatic number (max degree + 1)"""
        if not vertices:
            return 0
        
        max_degree = max(len(graph[v]) for v in vertices)
        return max_degree + 1
    
    def _compute_distance_matrix(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> Dict[int, Dict[int, int]]:
        """Compute all-pairs shortest path distances"""
        distances = {}
        
        for start in vertices:
            distances[start] = self._bfs_distances(graph, start)
        
        return distances
    
    def _bfs_distances(self, graph: Dict[int, Set[int]], start: int) -> Dict[int, int]:
        """BFS distances from start vertex"""
        distances = {start: 0}
        queue = [start]
        
        while queue:
            v = queue.pop(0)
            
            for u in graph[v]:
                if u not in distances:
                    distances[u] = distances[v] + 1
                    queue.append(u)
        
        return distances
    
    def _partition_by_degree(self, graph: Dict[int, Set[int]], vertices: Set[int]) -> List[List[int]]:
        """Partition vertices by degree"""
        degree_classes = defaultdict(list)
        
        for v in vertices:
            degree = len(graph[v])
            degree_classes[degree].append(v)
        
        return list(degree_classes.values())
    
    def _backtrack_mapping(self, classes1: List[List[int]], classes2: List[List[int]], 
                          mapping: Dict[int, int], class_idx: int) -> Optional[Dict[int, int]]:
        """Backtrack to find valid mapping"""
        if class_idx >= len(classes1):
            # Check if complete mapping is valid
            if self._is_valid_complete_mapping(mapping):
                return mapping
            return None
        
        class1 = classes1[class_idx]
        class2 = classes2[class_idx]
        
        if len(class1) != len(class2):
            return None
        
        # Try all permutations of class2 for class1
        for perm in itertools.permutations(class2):
            new_mapping = mapping.copy()
            for i, v1 in enumerate(class1):
                new_mapping[v1] = perm[i]
            
            # Check partial mapping validity
            if self._is_valid_partial_mapping(new_mapping, class_idx + 1, classes1):
                result = self._backtrack_mapping(classes1, classes2, new_mapping, class_idx + 1)
                if result:
                    return result
        
        return None
    
    def _is_valid_complete_mapping(self, mapping: Dict[int, int]) -> bool:
        """Check if complete mapping is valid isomorphism"""
        for u in self.vertices1:
            for v in self.graph1[u]:
                if mapping[v] not in self.graph2[mapping[u]]:
                    return False
        return True
    
    def _is_valid_partial_mapping(self, mapping: Dict[int, int], 
                                 processed_classes: int, classes1: List[List[int]]) -> bool:
        """Check if partial mapping is consistent"""
        mapped_vertices = set()
        for i in range(processed_classes):
            mapped_vertices.update(classes1[i])
        
        for u in mapped_vertices:
            for v in self.graph1[u]:
                if v in mapped_vertices:
                    if mapping[v] not in self.graph2[mapping[u]]:
                        return False
        
        return True

def test_advanced_isomorphism():
    """Test advanced graph isomorphism algorithms"""
    print("=== Testing Advanced Graph Isomorphism ===")
    
    # Test cases
    test_cases = [
        # Isomorphic graphs
        ([(0,1),(1,2),(2,3),(3,0)], [(0,1),(1,2),(2,3),(3,0)], True, "Same square"),
        ([(0,1),(1,2),(2,3),(3,0)], [(0,2),(2,1),(1,3),(3,0)], True, "Isomorphic squares"),
        
        # Non-isomorphic graphs
        ([(0,1),(1,2),(2,0)], [(0,1),(1,2),(2,3),(3,0)], False, "Triangle vs Square"),
        ([(0,1),(0,2),(1,2)], [(0,1),(1,2),(2,3)], False, "Triangle vs Path"),
    ]
    
    algorithms = [
        ("Weisfeiler-Lehman", lambda solver: solver.are_isomorphic_weisfeiler_lehman()),
        ("Invariant-based", lambda solver: solver.are_isomorphic_invariant_based()),
        ("Canonical Labeling", lambda solver: solver.are_isomorphic_canonical_labeling()),
    ]
    
    for edges1, edges2, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        solver = GraphIsomorphismAdvanced()
        solver.set_graphs(edges1, edges2)
        
        print(f"Graph 1: {edges1}")
        print(f"Graph 2: {edges2}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(solver)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Isomorphic: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")
        
        # Test mapping finding
        try:
            mapping = solver.find_isomorphism_mapping_advanced()
            if mapping:
                print(f"Mapping found: {mapping}")
            else:
                print("No mapping found")
        except Exception as e:
            print(f"Mapping ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_advanced_isomorphism()

"""
Advanced Graph Isomorphism demonstrates sophisticated techniques
for one of the most challenging problems in computational complexity,
combining theoretical algorithms with practical heuristics.
"""
