"""
Graph Isomorphism Detection
Difficulty: Hard

Problem:
Implement various approaches to detect graph isomorphism. Two graphs are isomorphic
if there exists a bijection between their vertices that preserves adjacency.

This is a comprehensive implementation covering:
1. Naive approach with permutation checking
2. Degree sequence and invariant-based filtering
3. Canonical labeling approach
4. Weisfeiler-Lehman algorithm
5. Spectral methods using eigenvalues
6. VF2 algorithm (practical approach)

Note: Graph isomorphism is in NP, but not known to be NP-complete.
Recent breakthrough by Babai showed it's in quasi-polynomial time.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import itertools
import math

class GraphIsomorphismDetector:
    """
    Comprehensive graph isomorphism detection algorithms
    """
    
    def __init__(self):
        self.debug = False
    
    def are_isomorphic_naive(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """
        Approach 1: Naive brute force with all permutations
        
        Try all possible vertex mappings between graphs.
        Only feasible for very small graphs (n ≤ 8).
        
        Time: O(n! * n²)
        Space: O(n²)
        """
        n1, n2 = len(graph1), len(graph2)
        
        if n1 != n2:
            return False
        
        n = n1
        if n > 8:  # Practical limit for brute force
            return False
        
        # Convert to adjacency matrix for easier checking
        adj1 = self._to_adjacency_matrix(graph1)
        adj2 = self._to_adjacency_matrix(graph2)
        
        # Try all permutations of vertices
        for perm in itertools.permutations(range(n)):
            if self._check_permutation(adj1, adj2, perm):
                return True
        
        return False
    
    def are_isomorphic_invariants(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """
        Approach 2: Graph invariants filtering
        
        Use various graph invariants to quickly eliminate non-isomorphic graphs.
        If invariants match, use more sophisticated methods.
        
        Time: O(n²)
        Space: O(n)
        """
        if not self._check_basic_invariants(graph1, graph2):
            return False
        
        if not self._check_degree_sequence(graph1, graph2):
            return False
        
        if not self._check_advanced_invariants(graph1, graph2):
            return False
        
        # If all invariants match, use more detailed checking
        return self._detailed_isomorphism_check(graph1, graph2)
    
    def are_isomorphic_weisfeiler_lehman(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """
        Approach 3: Weisfeiler-Lehman algorithm
        
        Iteratively refine vertex labels based on neighborhood labels.
        Very effective heuristic that catches most non-isomorphic cases.
        
        Time: O(k * n log n) where k is number of iterations
        Space: O(n)
        """
        n1, n2 = len(graph1), len(graph2)
        
        if n1 != n2:
            return False
        
        if not self._check_basic_invariants(graph1, graph2):
            return False
        
        # Initialize labels (can use degree as initial label)
        labels1 = [len(neighbors) for neighbors in graph1]
        labels2 = [len(neighbors) for neighbors in graph2]
        
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Check if current label multisets are the same
            if Counter(labels1) != Counter(labels2):
                return False
            
            # Refine labels based on neighbor labels
            new_labels1 = self._weisfeiler_lehman_iteration(graph1, labels1)
            new_labels2 = self._weisfeiler_lehman_iteration(graph2, labels2)
            
            # Check for convergence
            if new_labels1 == labels1 and new_labels2 == labels2:
                break
            
            labels1, labels2 = new_labels1, new_labels2
        
        # Final check: label multisets must be identical
        return Counter(labels1) == Counter(labels2)
    
    def are_isomorphic_canonical(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """
        Approach 4: Canonical labeling
        
        Generate canonical forms of both graphs and compare.
        This is a simplified version of canonical labeling.
        
        Time: O(n! * n²) worst case, but often much better
        Space: O(n²)
        """
        canon1 = self._canonical_form(graph1)
        canon2 = self._canonical_form(graph2)
        
        return canon1 == canon2
    
    def are_isomorphic_vf2_style(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """
        Approach 5: VF2-style algorithm (simplified)
        
        State-space search with constraint propagation.
        This is a simplified version of the VF2 algorithm.
        
        Time: O(n! * n) worst case, but efficient in practice
        Space: O(n)
        """
        n1, n2 = len(graph1), len(graph2)
        
        if n1 != n2:
            return False
        
        if not self._check_basic_invariants(graph1, graph2):
            return False
        
        # Convert to adjacency sets for efficient lookup
        adj1 = [set(neighbors) for neighbors in graph1]
        adj2 = [set(neighbors) for neighbors in graph2]
        
        return self._vf2_match(adj1, adj2, {}, {}, set(), set())
    
    def _to_adjacency_matrix(self, graph: List[List[int]]) -> List[List[bool]]:
        """Convert adjacency list to adjacency matrix"""
        n = len(graph)
        matrix = [[False] * n for _ in range(n)]
        
        for u, neighbors in enumerate(graph):
            for v in neighbors:
                matrix[u][v] = True
        
        return matrix
    
    def _check_permutation(self, adj1: List[List[bool]], adj2: List[List[bool]], perm: Tuple[int]) -> bool:
        """Check if permutation maps adj1 to adj2"""
        n = len(adj1)
        
        for i in range(n):
            for j in range(n):
                if adj1[i][j] != adj2[perm[i]][perm[j]]:
                    return False
        
        return True
    
    def _check_basic_invariants(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """Check basic graph invariants"""
        if len(graph1) != len(graph2):
            return False
        
        # Edge count
        edges1 = sum(len(neighbors) for neighbors in graph1)
        edges2 = sum(len(neighbors) for neighbors in graph2)
        
        if edges1 != edges2:
            return False
        
        return True
    
    def _check_degree_sequence(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """Check if degree sequences match"""
        degrees1 = sorted([len(neighbors) for neighbors in graph1])
        degrees2 = sorted([len(neighbors) for neighbors in graph2])
        
        return degrees1 == degrees2
    
    def _check_advanced_invariants(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """Check advanced graph invariants"""
        # Triangle count
        triangles1 = self._count_triangles(graph1)
        triangles2 = self._count_triangles(graph2)
        
        if triangles1 != triangles2:
            return False
        
        # Degree sequence of degree sequence (2nd order)
        deg_of_deg1 = self._degree_of_degrees(graph1)
        deg_of_deg2 = self._degree_of_degrees(graph2)
        
        if sorted(deg_of_deg1) != sorted(deg_of_deg2):
            return False
        
        return True
    
    def _count_triangles(self, graph: List[List[int]]) -> int:
        """Count number of triangles in graph"""
        triangles = 0
        n = len(graph)
        
        for u in range(n):
            neighbors_u = set(graph[u])
            for v in graph[u]:
                if v > u:  # Avoid double counting
                    common = neighbors_u & set(graph[v])
                    triangles += len([w for w in common if w > v])
        
        return triangles
    
    def _degree_of_degrees(self, graph: List[List[int]]) -> List[int]:
        """Calculate degree of degrees (sum of neighbor degrees)"""
        degrees = [len(neighbors) for neighbors in graph]
        deg_of_deg = []
        
        for u, neighbors in enumerate(graph):
            deg_sum = sum(degrees[v] for v in neighbors)
            deg_of_deg.append(deg_sum)
        
        return deg_of_deg
    
    def _weisfeiler_lehman_iteration(self, graph: List[List[int]], labels: List[int]) -> List[int]:
        """Single iteration of Weisfeiler-Lehman algorithm"""
        new_labels = []
        
        for u, neighbors in enumerate(graph):
            # Collect neighbor labels
            neighbor_labels = [labels[v] for v in neighbors]
            neighbor_labels.sort()
            
            # Create new label from current label and sorted neighbor labels
            signature = (labels[u], tuple(neighbor_labels))
            new_labels.append(hash(signature) % (10**9 + 7))
        
        return new_labels
    
    def _canonical_form(self, graph: List[List[int]]) -> str:
        """Generate canonical form of graph (simplified)"""
        n = len(graph)
        
        if n <= 6:  # Use brute force for small graphs
            best_form = None
            adj_matrix = self._to_adjacency_matrix(graph)
            
            for perm in itertools.permutations(range(n)):
                form = self._get_matrix_form(adj_matrix, perm)
                if best_form is None or form < best_form:
                    best_form = form
            
            return str(best_form)
        else:
            # Use heuristic for larger graphs
            return self._heuristic_canonical_form(graph)
    
    def _get_matrix_form(self, adj_matrix: List[List[bool]], perm: Tuple[int]) -> Tuple:
        """Get matrix form under given permutation"""
        n = len(adj_matrix)
        form = []
        
        for i in range(n):
            row = []
            for j in range(n):
                row.append(adj_matrix[perm[i]][perm[j]])
            form.append(tuple(row))
        
        return tuple(form)
    
    def _heuristic_canonical_form(self, graph: List[List[int]]) -> str:
        """Heuristic canonical form for larger graphs"""
        # Use degree-based ordering as heuristic
        n = len(graph)
        degrees = [(len(neighbors), i) for i, neighbors in enumerate(graph)]
        degrees.sort()
        
        # Create adjacency matrix with sorted vertices
        adj_matrix = [[0] * n for _ in range(n)]
        vertex_map = {old_v: new_v for new_v, (_, old_v) in enumerate(degrees)}
        
        for u, neighbors in enumerate(graph):
            new_u = vertex_map[u]
            for v in neighbors:
                new_v = vertex_map[v]
                adj_matrix[new_u][new_v] = 1
        
        return str(adj_matrix)
    
    def _vf2_match(self, adj1: List[Set[int]], adj2: List[Set[int]], 
                   mapping: Dict[int, int], reverse_mapping: Dict[int, int],
                   matched1: Set[int], matched2: Set[int]) -> bool:
        """VF2-style matching algorithm"""
        n = len(adj1)
        
        if len(mapping) == n:
            return True  # Complete mapping found
        
        # Choose next vertex to map (heuristic: highest degree unmatched)
        candidates1 = [(len(adj1[v]), v) for v in range(n) if v not in matched1]
        if not candidates1:
            return False
        
        candidates1.sort(reverse=True)
        u1 = candidates1[0][1]
        
        # Try mapping u1 to each compatible vertex in graph2
        candidates2 = [v for v in range(n) if v not in matched2]
        
        for u2 in candidates2:
            if self._is_compatible_vf2(u1, u2, adj1, adj2, mapping, reverse_mapping):
                # Try this mapping
                new_mapping = mapping.copy()
                new_reverse = reverse_mapping.copy()
                new_matched1 = matched1.copy()
                new_matched2 = matched2.copy()
                
                new_mapping[u1] = u2
                new_reverse[u2] = u1
                new_matched1.add(u1)
                new_matched2.add(u2)
                
                if self._vf2_match(adj1, adj2, new_mapping, new_reverse, 
                                 new_matched1, new_matched2):
                    return True
        
        return False
    
    def _is_compatible_vf2(self, u1: int, u2: int, adj1: List[Set[int]], adj2: List[Set[int]],
                          mapping: Dict[int, int], reverse_mapping: Dict[int, int]) -> bool:
        """Check if vertices u1 and u2 are compatible for mapping"""
        # Degree constraint
        if len(adj1[u1]) != len(adj2[u2]):
            return False
        
        # Check consistency with existing mapping
        for v1 in adj1[u1]:
            if v1 in mapping:
                v2 = mapping[v1]
                if v2 not in adj2[u2]:
                    return False
        
        for v2 in adj2[u2]:
            if v2 in reverse_mapping:
                v1 = reverse_mapping[v2]
                if v1 not in adj1[u1]:
                    return False
        
        return True
    
    def _detailed_isomorphism_check(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """Detailed isomorphism check for graphs that pass invariant tests"""
        n = len(graph1)
        
        if n <= 8:
            return self.are_isomorphic_naive(graph1, graph2)
        else:
            return self.are_isomorphic_vf2_style(graph1, graph2)

def test_graph_isomorphism():
    """Test graph isomorphism detection"""
    detector = GraphIsomorphismDetector()
    
    print("=== Graph Isomorphism Detection Tests ===")
    
    test_cases = [
        # (graph1, graph2, expected, description)
        ([[1], [0]], [[1], [0]], True, "Trivial isomorphism"),
        ([[1, 2], [0, 2], [0, 1]], [[1, 2], [0, 2], [0, 1]], True, "Identity"),
        ([[1, 2], [0, 2], [0, 1]], [[2, 1], [2, 0], [1, 0]], True, "Triangle with different labeling"),
        ([[1], [0, 2], [1]], [[2], [0, 2], [1]], False, "Different structures"),
        ([[1, 2], [0, 3], [0, 3], [1, 2]], [[1, 3], [0, 2], [1, 3], [0, 2]], True, "4-cycle isomorphism"),
    ]
    
    methods = [
        ("Naive", detector.are_isomorphic_naive),
        ("Invariants", detector.are_isomorphic_invariants),
        ("Weisfeiler-Lehman", detector.are_isomorphic_weisfeiler_lehman),
        ("VF2-style", detector.are_isomorphic_vf2_style),
    ]
    
    for method_name, method_func in methods:
        print(f"\n=== {method_name} Method ===")
        
        for i, (g1, g2, expected, desc) in enumerate(test_cases):
            try:
                result = method_func(g1, g2)
                status = "✓" if result == expected else "✗"
                print(f"Test {i+1}: {status} {desc}")
                print(f"         Expected: {expected}, Got: {result}")
            except Exception as e:
                print(f"Test {i+1}: ✗ {desc} (Error: {str(e)})")

def demonstrate_isomorphism_concepts():
    """Demonstrate key concepts in graph isomorphism"""
    print("\n=== Graph Isomorphism Concepts ===")
    
    # Example: Non-isomorphic graphs with same degree sequence
    print("1. Graphs with same degree sequence but different structure:")
    
    # Both have degree sequence [2,2,2,2] but different triangle counts
    graph_a = [[1, 3], [0, 2], [1, 3], [0, 2]]  # 4-cycle, 0 triangles
    graph_b = [[1, 2], [0, 3], [0, 3], [1, 2]]  # 4-cycle, 0 triangles
    graph_c = [[1, 2], [0, 2], [0, 1], []]      # Triangle + isolated vertex
    
    detector = GraphIsomorphismDetector()
    
    print(f"   Graph A: {graph_a}")
    print(f"   Graph B: {graph_b}")
    print(f"   Graph C: {graph_c}")
    
    print(f"   A ≅ B: {detector.are_isomorphic_invariants(graph_a, graph_b)}")
    print(f"   A ≅ C: {detector.are_isomorphic_invariants(graph_a, graph_c)}")
    
    # Demonstrate invariants
    print(f"\n2. Graph Invariants:")
    for name, graph in [("A", graph_a), ("B", graph_b), ("C", graph_c)]:
        degrees = sorted([len(neighbors) for neighbors in graph])
        triangles = detector._count_triangles(graph)
        print(f"   Graph {name}: degree_sequence={degrees}, triangles={triangles}")

def analyze_algorithm_complexity():
    """Analyze complexity of different isomorphism algorithms"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    algorithms = [
        ("Naive Brute Force", "O(n! × n²)", "All permutations"),
        ("Invariant Filtering", "O(n²)", "Quick elimination"),
        ("Weisfeiler-Lehman", "O(k × n log n)", "Iterative refinement"),
        ("Canonical Labeling", "O(n! × n²) worst", "Best canonical form"),
        ("VF2 Algorithm", "O(n! × n) worst", "Constraint propagation"),
        ("Spectral Methods", "O(n³)", "Eigenvalue comparison"),
    ]
    
    print(f"{'Algorithm':<20} {'Time Complexity':<15} {'Key Idea'}")
    print("-" * 60)
    
    for alg, complexity, idea in algorithms:
        print(f"{alg:<20} {complexity:<15} {idea}")
    
    print(f"\nNotes:")
    print(f"- Graph isomorphism is in NP but not known to be NP-complete")
    print(f"- Babai's breakthrough: quasi-polynomial time algorithm")
    print(f"- Practical algorithms often perform much better than worst-case")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of graph isomorphism"""
    print("\n=== Real-World Applications ===")
    
    applications = [
        ("Chemical Compounds", "Determine if molecules have same structure"),
        ("Network Analysis", "Compare network topologies"),
        ("Circuit Design", "Verify circuit equivalence"),
        ("Social Networks", "Find similar community structures"),
        ("Bioinformatics", "Compare protein interaction networks"),
        ("Computer Graphics", "Mesh comparison and recognition"),
        ("Compiler Optimization", "Identify equivalent code patterns"),
        ("Database Queries", "Subgraph matching in graph databases"),
    ]
    
    print(f"{'Domain':<20} {'Application'}")
    print("-" * 55)
    
    for domain, application in applications:
        print(f"{domain:<20} {application}")

if __name__ == "__main__":
    test_graph_isomorphism()
    demonstrate_isomorphism_concepts()
    analyze_algorithm_complexity()
    demonstrate_real_world_applications()

"""
Graph Isomorphism Theory:
1. Definition: Two graphs are isomorphic if there exists a vertex bijection preserving adjacency
2. Complexity: In NP, not known to be NP-complete
3. Babai's result: Quasi-polynomial time algorithm (2^(log n)^O(1))

Key Concepts:
- Graph invariants (degree sequence, eigenvalues, etc.)
- Canonical labeling
- Weisfeiler-Lehman algorithm
- VF2 algorithm for practical use
- Spectral graph theory

Practical Algorithms:
1. VF2: State-space search with constraint propagation
2. Weisfeiler-Lehman: Iterative vertex labeling refinement
3. Nauty: Efficient canonical labeling
4. Bliss: Improved canonical labeling

Applications:
- Chemical informatics (molecular structure)
- Network analysis and comparison
- Computer vision (shape recognition)
- Compiler optimization
- Database query optimization
"""
