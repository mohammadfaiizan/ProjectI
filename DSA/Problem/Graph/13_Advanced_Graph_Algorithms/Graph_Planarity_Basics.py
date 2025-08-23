"""
Graph Planarity Basics - Multiple Approaches
Difficulty: Easy

A graph is planar if it can be drawn on a plane without edge crossings.
This file implements basic planarity testing algorithms and related concepts.

Key Concepts:
1. Planar Graph Properties
2. Kuratowski's Theorem
3. Euler's Formula for Planar Graphs
4. Basic Planarity Tests
5. Planar Graph Recognition
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import math

class GraphPlanarityBasics:
    """Basic planarity testing and analysis"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.edges = []
    
    def add_edge(self, u: int, v: int):
        """Add edge to graph"""
        self.graph[u].append(v)
        self.graph[v].append(u)
        self.vertices.add(u)
        self.vertices.add(v)
        if (u, v) not in self.edges and (v, u) not in self.edges:
            self.edges.append((u, v))
    
    def is_planar_euler_formula(self) -> bool:
        """
        Approach 1: Euler's Formula Check
        
        For connected planar graph: V - E + F = 2
        For planar graph: E ≤ 3V - 6 (V ≥ 3)
        
        Time: O(1), Space: O(1)
        """
        V = len(self.vertices)
        E = len(self.edges)
        
        if V == 0:
            return True
        
        if V <= 2:
            return True
        
        # Necessary condition: E ≤ 3V - 6
        if E > 3 * V - 6:
            return False
        
        # For bipartite graphs: E ≤ 2V - 4
        if self._is_bipartite() and E > 2 * V - 4:
            return False
        
        return True
    
    def is_planar_kuratowski_basic(self) -> bool:
        """
        Approach 2: Basic Kuratowski Check
        
        Check for K5 and K3,3 subgraphs (simplified version).
        
        Time: O(V^5), Space: O(V)
        """
        V = len(self.vertices)
        
        if V < 5:
            return True
        
        # Check for K5 (complete graph on 5 vertices)
        if self._has_k5_subgraph():
            return False
        
        if V >= 6:
            # Check for K3,3 (complete bipartite graph)
            if self._has_k33_subgraph():
                return False
        
        return True
    
    def is_planar_dfs_based(self) -> bool:
        """
        Approach 3: DFS-Based Planarity Test
        
        Simplified planarity test using DFS properties.
        
        Time: O(V + E), Space: O(V)
        """
        if not self.vertices:
            return True
        
        # Check basic necessary conditions
        if not self.is_planar_euler_formula():
            return False
        
        # Check connectivity and cycles
        if not self._is_connected():
            # For disconnected graphs, check each component
            components = self._get_connected_components()
            for component in components:
                subgraph = self._extract_subgraph(component)
                if not subgraph.is_planar_euler_formula():
                    return False
        
        return True
    
    def is_planar_degree_based(self) -> bool:
        """
        Approach 4: Degree-Based Heuristic
        
        Use degree properties for quick planarity assessment.
        
        Time: O(V), Space: O(1)
        """
        if len(self.vertices) <= 4:
            return True
        
        # Check maximum degree
        max_degree = max(len(self.graph[v]) for v in self.vertices)
        
        # In planar graphs, there's always a vertex with degree ≤ 5
        if max_degree > 5:
            degrees = [len(self.graph[v]) for v in self.vertices]
            if min(degrees) > 5:
                return False
        
        return self.is_planar_euler_formula()
    
    def count_faces_euler(self) -> int:
        """
        Approach 5: Face Counting using Euler's Formula
        
        Calculate number of faces using V - E + F = 2.
        
        Time: O(1), Space: O(1)
        """
        if not self._is_connected():
            return -1  # Formula applies to connected graphs
        
        V = len(self.vertices)
        E = len(self.edges)
        
        if V == 0:
            return 1  # Empty graph has 1 face (outer face)
        
        # F = 2 - V + E
        return 2 - V + E
    
    def get_planarity_properties(self) -> Dict:
        """
        Approach 6: Comprehensive Planarity Analysis
        
        Return various planarity-related properties.
        
        Time: O(V + E), Space: O(V)
        """
        V = len(self.vertices)
        E = len(self.edges)
        
        properties = {
            'vertices': V,
            'edges': E,
            'is_connected': self._is_connected(),
            'is_bipartite': self._is_bipartite(),
            'max_degree': max(len(self.graph[v]) for v in self.vertices) if V > 0 else 0,
            'min_degree': min(len(self.graph[v]) for v in self.vertices) if V > 0 else 0,
            'euler_formula_satisfied': self.is_planar_euler_formula(),
            'estimated_faces': self.count_faces_euler() if self._is_connected() else -1,
            'density': (2 * E) / (V * (V - 1)) if V > 1 else 0,
        }
        
        # Planarity assessment
        properties['likely_planar'] = (
            properties['euler_formula_satisfied'] and
            not self._has_obvious_non_planarity()
        )
        
        return properties
    
    def _is_bipartite(self) -> bool:
        """Check if graph is bipartite using BFS coloring"""
        if not self.vertices:
            return True
        
        color = {}
        
        for start in self.vertices:
            if start in color:
                continue
            
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in self.graph[node]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
        
        return True
    
    def _is_connected(self) -> bool:
        """Check if graph is connected"""
        if not self.vertices:
            return True
        
        start = next(iter(self.vertices))
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            node = queue.popleft()
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self.vertices)
    
    def _get_connected_components(self) -> List[Set[int]]:
        """Get all connected components"""
        visited = set()
        components = []
        
        for vertex in self.vertices:
            if vertex not in visited:
                component = set()
                queue = deque([vertex])
                
                while queue:
                    node = queue.popleft()
                    if node not in visited:
                        visited.add(node)
                        component.add(node)
                        
                        for neighbor in self.graph[node]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def _extract_subgraph(self, vertices: Set[int]) -> 'GraphPlanarityBasics':
        """Extract subgraph induced by given vertices"""
        subgraph = GraphPlanarityBasics()
        
        for u in vertices:
            for v in self.graph[u]:
                if v in vertices and u < v:  # Avoid duplicate edges
                    subgraph.add_edge(u, v)
        
        return subgraph
    
    def _has_k5_subgraph(self) -> bool:
        """Check for K5 subgraph (simplified)"""
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        
        if n < 5:
            return False
        
        # Check all combinations of 5 vertices
        from itertools import combinations
        
        for combo in combinations(vertices_list, 5):
            if self._is_complete_subgraph(combo):
                return True
        
        return False
    
    def _has_k33_subgraph(self) -> bool:
        """Check for K3,3 subgraph (simplified)"""
        if not self._is_bipartite():
            return False
        
        # For bipartite graph, check for K3,3
        vertices_list = list(self.vertices)
        n = len(vertices_list)
        
        if n < 6:
            return False
        
        # Get bipartition
        color = {}
        start = vertices_list[0]
        queue = deque([start])
        color[start] = 0
        
        while queue:
            node = queue.popleft()
            for neighbor in self.graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
        
        part1 = [v for v in self.vertices if color[v] == 0]
        part2 = [v for v in self.vertices if color[v] == 1]
        
        if len(part1) >= 3 and len(part2) >= 3:
            from itertools import combinations
            
            for combo1 in combinations(part1, 3):
                for combo2 in combinations(part2, 3):
                    if self._is_complete_bipartite_subgraph(combo1, combo2):
                        return True
        
        return False
    
    def _is_complete_subgraph(self, vertices: Tuple[int, ...]) -> bool:
        """Check if given vertices form a complete subgraph"""
        for i, u in enumerate(vertices):
            for j, v in enumerate(vertices):
                if i < j and v not in self.graph[u]:
                    return False
        return True
    
    def _is_complete_bipartite_subgraph(self, part1: Tuple[int, ...], part2: Tuple[int, ...]) -> bool:
        """Check if given bipartition forms complete bipartite subgraph"""
        for u in part1:
            for v in part2:
                if v not in self.graph[u]:
                    return False
        return True
    
    def _has_obvious_non_planarity(self) -> bool:
        """Check for obvious non-planarity indicators"""
        V = len(self.vertices)
        E = len(self.edges)
        
        if V <= 4:
            return False
        
        # Very dense graphs are likely non-planar
        if E > 2.5 * V:
            return True
        
        # High minimum degree suggests non-planarity
        if V > 0:
            min_degree = min(len(self.graph[v]) for v in self.vertices)
            if min_degree > 5:
                return True
        
        return False

def test_graph_planarity():
    """Test graph planarity algorithms"""
    print("=== Testing Graph Planarity Basics ===")
    
    # Test cases
    test_graphs = [
        # Planar graphs
        ([(0, 1), (1, 2), (2, 3), (3, 0)], True, "Square (C4)"),
        ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], False, "K4 - should be planar but complex"),
        ([(0, 1), (1, 2), (2, 0)], True, "Triangle (C3)"),
        ([(0, 1), (1, 2)], True, "Path"),
        ([], True, "Empty graph"),
        
        # Non-planar graphs (simplified detection)
        ([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)], False, "K5"),
    ]
    
    algorithms = [
        ("Euler Formula", lambda g: g.is_planar_euler_formula()),
        ("Kuratowski Basic", lambda g: g.is_planar_kuratowski_basic()),
        ("DFS Based", lambda g: g.is_planar_dfs_based()),
        ("Degree Based", lambda g: g.is_planar_degree_based()),
    ]
    
    for edges, expected_planar, description in test_graphs:
        print(f"\n--- {description} ---")
        
        graph = GraphPlanarityBasics()
        for u, v in edges:
            graph.add_edge(u, v)
        
        print(f"Edges: {edges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(graph)
                # Note: Our basic algorithms may not be 100% accurate for complex cases
                print(f"{alg_name:15} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")
        
        # Show properties
        properties = graph.get_planarity_properties()
        print(f"Properties: V={properties['vertices']}, E={properties['edges']}, "
              f"Connected={properties['is_connected']}, Likely Planar={properties['likely_planar']}")

def demonstrate_planarity_concepts():
    """Demonstrate planarity concepts"""
    print("\n=== Planarity Concepts Demo ===")
    
    print("Key Planarity Properties:")
    print("1. Euler's Formula: V - E + F = 2 (connected planar graphs)")
    print("2. Edge Bound: E ≤ 3V - 6 (V ≥ 3)")
    print("3. Bipartite Bound: E ≤ 2V - 4 (bipartite planar graphs)")
    print("4. Kuratowski's Theorem: No K5 or K3,3 subdivisions")
    
    print("\nFamous Non-Planar Graphs:")
    print("• K5 (complete graph on 5 vertices)")
    print("• K3,3 (complete bipartite graph 3,3)")
    print("• Petersen graph")
    print("• Any graph containing K5 or K3,3 subdivision")
    
    print("\nApplications:")
    print("• Circuit board design")
    print("• Map coloring")
    print("• Network layout")
    print("• Graph drawing algorithms")

def analyze_planarity_complexity():
    """Analyze complexity of planarity algorithms"""
    print("\n=== Planarity Algorithm Complexity ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Euler Formula Check:**")
    print("   • Time: O(1)")
    print("   • Space: O(1)")
    print("   • Pros: Very fast, necessary condition")
    print("   • Cons: Not sufficient, many false positives")
    
    print("\n2. **Basic Kuratowski:**")
    print("   • Time: O(V^5) - checking all 5-vertex subsets")
    print("   • Space: O(V)")
    print("   • Pros: Based on fundamental theorem")
    print("   • Cons: Exponential time, incomplete implementation")
    
    print("\n3. **DFS-Based:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V)")
    print("   • Pros: Linear time, good heuristics")
    print("   • Cons: Not complete planarity test")
    
    print("\n4. **Advanced Algorithms (not implemented):**")
    print("   • Hopcroft-Tarjan: O(V) time")
    print("   • Boyer-Myrvold: O(V) time")
    print("   • Path addition methods")
    
    print("\nNote: Complete planarity testing is complex and requires")
    print("sophisticated algorithms like Hopcroft-Tarjan or Boyer-Myrvold.")

if __name__ == "__main__":
    test_graph_planarity()
    demonstrate_planarity_concepts()
    analyze_planarity_complexity()

"""
Graph Planarity Basics - Key Insights:

1. **Planarity Fundamentals:**
   - Planar graphs can be drawn without edge crossings
   - Euler's formula: V - E + F = 2 for connected planar graphs
   - Edge bounds: E ≤ 3V - 6 for simple planar graphs
   - Kuratowski's theorem: No K5 or K3,3 subdivisions

2. **Testing Approaches:**
   - Necessary conditions: Euler formula, degree bounds
   - Structural tests: Kuratowski subgraph detection
   - Heuristic methods: Degree analysis, connectivity
   - Complete algorithms: Hopcroft-Tarjan (not implemented)

3. **Complexity Considerations:**
   - Basic tests: O(1) to O(V + E)
   - Complete planarity: O(V) with advanced algorithms
   - Subgraph detection: Exponential in general
   - Practical heuristics often sufficient

4. **Applications:**
   - Circuit design and VLSI layout
   - Graph drawing and visualization
   - Network topology analysis
   - Map and geographic applications

5. **Implementation Notes:**
   - This provides basic planarity concepts
   - Complete planarity testing requires advanced algorithms
   - Useful for educational purposes and simple cases
   - Production systems need Hopcroft-Tarjan or similar

The basic planarity tests provide foundation understanding
while highlighting the complexity of complete planarity testing.
"""
