"""
Articulation Points and Bridges
Difficulty: Medium

Articulation Points (Cut Vertices): Vertices whose removal increases the number of connected components.
Bridges (Cut Edges): Edges whose removal increases the number of connected components.

These are fundamental concepts in graph connectivity and network reliability analysis.

Key Algorithms:
1. Tarjan's Algorithm for finding articulation points and bridges
2. Naive approaches for educational understanding
3. Applications in network reliability and robustness analysis

Applications:
- Network reliability and fault tolerance
- Infrastructure vulnerability assessment
- Social network analysis and community detection
- Circuit design and critical path analysis
- Transportation network robustness
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict

class ArticulationAnalysis:
    """Comprehensive articulation point and bridge analysis"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset state for new analysis"""
        self.time = 0
        self.discovery = {}
        self.low = {}
        self.parent = {}
        self.visited = set()
        self.articulation_points = set()
        self.bridges = []
    
    def find_articulation_points_and_bridges_tarjan(self, graph: Dict[int, List[int]]) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Approach 1: Tarjan's Algorithm (Optimal)
        
        Find both articulation points and bridges in single DFS traversal.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset()
        
        def tarjan_dfs(u):
            """DFS for Tarjan's algorithm"""
            children = 0
            self.visited.add(u)
            self.discovery[u] = self.low[u] = self.time
            self.time += 1
            
            for v in graph.get(u, []):
                if v not in self.visited:
                    children += 1
                    self.parent[v] = u
                    tarjan_dfs(v)
                    
                    # Update low value
                    self.low[u] = min(self.low[u], self.low[v])
                    
                    # Check for articulation point
                    if self.parent.get(u) is None and children > 1:
                        # Root with multiple children
                        self.articulation_points.add(u)
                    elif self.parent.get(u) is not None and self.low[v] >= self.discovery[u]:
                        # Non-root with cutoff condition
                        self.articulation_points.add(u)
                    
                    # Check for bridge
                    if self.low[v] > self.discovery[u]:
                        self.bridges.append((min(u, v), max(u, v)))
                        
                elif v != self.parent.get(u):
                    # Back edge (not to parent)
                    self.low[u] = min(self.low[u], self.discovery[v])
        
        # Find all vertices
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        # Run DFS from all unvisited vertices
        for vertex in vertices:
            if vertex not in self.visited:
                tarjan_dfs(vertex)
        
        return list(self.articulation_points), self.bridges
    
    def find_articulation_points_naive(self, graph: Dict[int, List[int]]) -> List[int]:
        """
        Approach 2: Naive Articulation Point Finding
        
        Try removing each vertex and check if components increase.
        
        Time: O(V * (V + E))
        Space: O(V)
        """
        def count_components(excluded_vertex=None):
            """Count connected components excluding a vertex"""
            visited = set()
            components = 0
            
            def dfs(vertex):
                if vertex in visited or vertex == excluded_vertex:
                    return
                visited.add(vertex)
                for neighbor in graph.get(vertex, []):
                    dfs(neighbor)
            
            for vertex in graph:
                if vertex != excluded_vertex and vertex not in visited:
                    dfs(vertex)
                    components += 1
            
            return components
        
        original_components = count_components()
        articulation_points = []
        
        for vertex in graph:
            if count_components(vertex) > original_components:
                articulation_points.append(vertex)
        
        return articulation_points
    
    def find_bridges_naive(self, graph: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        """
        Approach 3: Naive Bridge Finding
        
        Try removing each edge and check if components increase.
        
        Time: O(E * (V + E))
        Space: O(V + E)
        """
        def count_components_without_edge(excluded_edge):
            """Count components without specific edge"""
            visited = set()
            components = 0
            
            def dfs(vertex):
                if vertex in visited:
                    return
                visited.add(vertex)
                
                for neighbor in graph.get(vertex, []):
                    edge = (min(vertex, neighbor), max(vertex, neighbor))
                    if edge != excluded_edge:
                        dfs(neighbor)
            
            for vertex in graph:
                if vertex not in visited:
                    dfs(vertex)
                    components += 1
            
            return components
        
        # Get all edges
        edges = set()
        for u in graph:
            for v in graph[u]:
                edge = (min(u, v), max(u, v))
                edges.add(edge)
        
        original_components = count_components_without_edge(None)
        bridges = []
        
        for edge in edges:
            if count_components_without_edge(edge) > original_components:
                bridges.append(edge)
        
        return bridges
    
    def find_articulation_points_iterative(self, graph: Dict[int, List[int]]) -> List[int]:
        """
        Approach 4: Iterative Tarjan's for Articulation Points
        
        Iterative implementation to avoid recursion limits.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset()
        articulation_points = set()
        
        def iterative_tarjan(start):
            """Iterative DFS for articulation points"""
            stack = [(start, 0, False)]  # (vertex, neighbor_index, returning)
            
            while stack:
                u, neighbor_idx, returning = stack.pop()
                
                if returning:
                    # Returning from recursive call
                    if neighbor_idx > 0:
                        neighbors = list(graph.get(u, []))
                        if neighbor_idx <= len(neighbors):
                            v = neighbors[neighbor_idx - 1]
                            if self.parent.get(v) == u:  # Tree edge
                                self.low[u] = min(self.low[u], self.low[v])
                                
                                # Check articulation point
                                if (self.parent.get(u) is None and 
                                    sum(1 for w in neighbors if self.parent.get(w) == u) > 1):
                                    articulation_points.add(u)
                                elif (self.parent.get(u) is not None and 
                                      self.low[v] >= self.discovery[u]):
                                    articulation_points.add(u)
                    
                    # Continue with next neighbor
                    neighbors = list(graph.get(u, []))
                    while neighbor_idx < len(neighbors):
                        v = neighbors[neighbor_idx]
                        neighbor_idx += 1
                        
                        if v not in self.visited:
                            self.parent[v] = u
                            stack.append((u, neighbor_idx, True))
                            stack.append((v, 0, False))
                            break
                        elif v != self.parent.get(u):
                            self.low[u] = min(self.low[u], self.discovery[v])
                else:
                    # First visit
                    if u in self.visited:
                        continue
                    
                    self.visited.add(u)
                    self.discovery[u] = self.low[u] = self.time
                    self.time += 1
                    
                    # Process neighbors
                    neighbors = list(graph.get(u, []))
                    if neighbors:
                        v = neighbors[0]
                        if v not in self.visited:
                            self.parent[v] = u
                            stack.append((u, 1, True))
                            stack.append((v, 0, False))
                        else:
                            if v != self.parent.get(u):
                                self.low[u] = min(self.low[u], self.discovery[v])
                            stack.append((u, 1, True))
        
        # Find all vertices
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        # Process all components
        for vertex in vertices:
            if vertex not in self.visited:
                iterative_tarjan(vertex)
        
        return list(articulation_points)
    
    def analyze_network_vulnerability(self, graph: Dict[int, List[int]]) -> Dict:
        """
        Approach 5: Comprehensive Network Vulnerability Analysis
        
        Analyze network robustness using articulation points and bridges.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        articulation_points, bridges = self.find_articulation_points_and_bridges_tarjan(graph)
        
        # Count components
        def count_components():
            visited = set()
            components = 0
            
            def dfs(vertex):
                if vertex in visited:
                    return
                visited.add(vertex)
                for neighbor in graph.get(vertex, []):
                    dfs(neighbor)
            
            for vertex in graph:
                if vertex not in visited:
                    dfs(vertex)
                    components += 1
            
            return components
        
        # Analyze connectivity
        total_vertices = len(set(v for v in graph.keys()) | 
                            set(v for neighbors in graph.values() for v in neighbors))
        total_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        
        # Calculate vulnerability metrics
        vulnerability_score = (len(articulation_points) * 2 + len(bridges)) / max(total_vertices, 1)
        
        # Find critical paths (paths through articulation points)
        critical_paths = []
        for ap in articulation_points:
            # Remove articulation point and find resulting components
            temp_graph = {u: [v for v in neighbors if v != ap] 
                         for u, neighbors in graph.items() if u != ap}
            
            # This would require more complex analysis for full implementation
        
        return {
            'articulation_points': articulation_points,
            'bridges': bridges,
            'total_vertices': total_vertices,
            'total_edges': total_edges,
            'components': count_components(),
            'vulnerability_score': vulnerability_score,
            'is_biconnected': len(articulation_points) == 0 and count_components() == 1,
            'is_bridge_connected': len(bridges) == total_vertices - count_components(),
            'robustness_level': 'High' if vulnerability_score < 0.1 else 
                              'Medium' if vulnerability_score < 0.3 else 'Low'
        }

def test_articulation_analysis():
    """Test all approaches with various graphs"""
    analyzer = ArticulationAnalysis()
    
    test_graphs = [
        # Simple bridge
        {0: [1], 1: [0, 2], 2: [1]},
        
        # Articulation point
        {0: [1], 1: [0, 2, 3], 2: [1], 3: [1]},
        
        # Complex graph
        {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4, 5], 4: [3, 5], 5: [3, 4]},
        
        # Cycle (no articulation points or bridges)
        {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]},
        
        # Tree (all internal vertices are articulation points, all edges are bridges)
        {0: [1], 1: [0, 2, 3], 2: [1], 3: [1, 4], 4: [3]},
    ]
    
    for i, graph in enumerate(test_graphs):
        print(f"\n=== Test Graph {i+1} ===")
        print(f"Graph: {graph}")
        
        # Test Tarjan's algorithm
        ap_tarjan, bridges_tarjan = analyzer.find_articulation_points_and_bridges_tarjan(graph)
        print(f"Tarjan's - Articulation points: {sorted(ap_tarjan)}")
        print(f"Tarjan's - Bridges: {sorted(bridges_tarjan)}")
        
        # Test naive approaches
        ap_naive = analyzer.find_articulation_points_naive(graph)
        bridges_naive = analyzer.find_bridges_naive(graph)
        print(f"Naive - Articulation points: {sorted(ap_naive)}")
        print(f"Naive - Bridges: {sorted(bridges_naive)}")
        
        # Verify consistency
        if sorted(ap_tarjan) == sorted(ap_naive) and sorted(bridges_tarjan) == sorted(bridges_naive):
            print("✓ All approaches agree")
        else:
            print("✗ Approaches disagree!")
        
        # Network analysis
        analysis = analyzer.analyze_network_vulnerability(graph)
        print(f"Vulnerability: {analysis['robustness_level']} (score: {analysis['vulnerability_score']:.2f})")

def demonstrate_tarjan_algorithm():
    """Demonstrate Tarjan's algorithm step by step"""
    print("\n=== Tarjan's Algorithm Demo ===")
    
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2, 4], 4: [3]}
    
    print(f"Graph: {graph}")
    print(f"Structure: 0-1-2 triangle connected to 3-4 edge via vertex 2")
    
    print(f"\nStep-by-step Tarjan's execution:")
    print(f"1. Start DFS from vertex 0")
    print(f"   discovery[0] = 0, low[0] = 0")
    
    print(f"2. Visit vertex 1 (child of 0)")
    print(f"   discovery[1] = 1, low[1] = 1")
    
    print(f"3. Visit vertex 2 (child of 1)")
    print(f"   discovery[2] = 2, low[2] = 2")
    
    print(f"4. Back edge from 2 to 0")
    print(f"   low[2] = min(low[2], discovery[0]) = min(2, 0) = 0")
    
    print(f"5. Return to 1, update low[1]")
    print(f"   low[1] = min(low[1], low[2]) = min(1, 0) = 0")
    
    print(f"6. Visit vertex 3 (child of 2)")
    print(f"   discovery[3] = 3, low[3] = 3")
    
    print(f"7. Visit vertex 4 (child of 3)")
    print(f"   discovery[4] = 4, low[4] = 4")
    
    print(f"8. Return to 3: low[4] = 4 > discovery[3] = 3")
    print(f"   → Edge (3,4) is a bridge!")
    
    print(f"9. Return to 2: low[3] = 3 >= discovery[2] = 2")
    print(f"   → Vertex 2 is an articulation point!")
    
    analyzer = ArticulationAnalysis()
    ap, bridges = analyzer.find_articulation_points_and_bridges_tarjan(graph)
    print(f"\nActual result:")
    print(f"  Articulation points: {ap}")
    print(f"  Bridges: {bridges}")

def demonstrate_network_applications():
    """Demonstrate real-world network applications"""
    print("\n=== Network Applications Demo ===")
    
    # Transportation network example
    transport_network = {
        'A': ['B', 'C'],      # City A connects to B and C
        'B': ['A', 'D'],      # City B connects to A and D
        'C': ['A', 'E'],      # City C connects to A and E
        'D': ['B', 'F'],      # City D connects to B and F
        'E': ['C', 'F'],      # City E connects to C and F
        'F': ['D', 'E']       # City F connects to D and E
    }
    
    # Convert to numeric for analysis
    city_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    numeric_network = {}
    for city, connections in transport_network.items():
        numeric_network[city_map[city]] = [city_map[conn] for conn in connections]
    
    print(f"Transportation Network:")
    for city, connections in transport_network.items():
        print(f"  {city} ↔ {connections}")
    
    analyzer = ArticulationAnalysis()
    analysis = analyzer.analyze_network_vulnerability(numeric_network)
    
    print(f"\nNetwork Analysis:")
    print(f"  Critical cities (articulation points): {[list(city_map.keys())[list(city_map.values()).index(ap)] for ap in analysis['articulation_points']]}")
    print(f"  Critical roads (bridges): {[(list(city_map.keys())[list(city_map.values()).index(u)], list(city_map.keys())[list(city_map.values()).index(v)]) for u, v in analysis['bridges']]}")
    print(f"  Robustness level: {analysis['robustness_level']}")
    print(f"  Vulnerability score: {analysis['vulnerability_score']:.2f}")

def analyze_connectivity_concepts():
    """Analyze connectivity concepts and properties"""
    print("\n=== Connectivity Concepts Analysis ===")
    
    print("Graph Connectivity Concepts:")
    
    print("\n1. **Articulation Points (Cut Vertices):**")
    print("   • Vertices whose removal increases connected components")
    print("   • Critical for network connectivity")
    print("   • Found using Tarjan's algorithm in O(V + E)")
    print("   • Root case: more than one child in DFS tree")
    print("   • Non-root case: low[v] ≥ discovery[u] for edge (u,v)")
    
    print("\n2. **Bridges (Cut Edges):**")
    print("   • Edges whose removal increases connected components")
    print("   • Critical connections in networks")
    print("   • Condition: low[v] > discovery[u] for edge (u,v)")
    print("   • No back edge from subtree of v to ancestors of u")
    
    print("\n3. **Biconnected Components:**")
    print("   • Maximal subgraphs with no articulation points")
    print("   • Any two vertices connected by two vertex-disjoint paths")
    print("   • Important for fault-tolerant network design")
    print("   • Can be found using articulation point analysis")
    
    print("\n4. **Bridge-Connected Components:**")
    print("   • Maximal subgraphs with no bridges")
    print("   • Any two vertices connected by two edge-disjoint paths")
    print("   • 2-edge-connected components")
    print("   • Important for communication reliability")
    
    print("\n5. **Network Robustness Metrics:**")
    print("   • Vertex connectivity: minimum vertices to remove for disconnection")
    print("   • Edge connectivity: minimum edges to remove for disconnection")
    print("   • Robustness coefficient: ratio of critical elements to total")
    print("   • Fault tolerance: ability to maintain connectivity under failures")

def demonstrate_practical_applications():
    """Demonstrate practical applications of articulation analysis"""
    print("\n=== Practical Applications ===")
    
    print("Articulation Point and Bridge Applications:")
    
    print("\n1. **Network Infrastructure:**")
    print("   • Internet backbone reliability analysis")
    print("   • Telecommunications network robustness")
    print("   • Power grid critical node identification")
    print("   • Transportation bottleneck detection")
    
    print("\n2. **Social Networks:**")
    print("   • Key influencer identification")
    print("   • Community bridge detection")
    print("   • Information flow bottlenecks")
    print("   • Social cohesion analysis")
    
    print("\n3. **Computer Networks:**")
    print("   • Router failure impact assessment")
    print("   • Network partition prevention")
    print("   • Redundancy planning")
    print("   • Load balancing optimization")
    
    print("\n4. **Biological Networks:**")
    print("   • Protein interaction critical nodes")
    print("   • Metabolic pathway bottlenecks")
    print("   • Gene regulatory network hubs")
    print("   • Ecosystem keystone species")
    
    print("\n5. **Circuit Design:**")
    print("   • Critical component identification")
    print("   • Fault tolerance analysis")
    print("   • Signal path redundancy")
    print("   • Reliability optimization")
    
    print("\nKey Benefits:")
    print("• Identify single points of failure")
    print("• Guide redundancy investments")
    print("• Optimize network robustness")
    print("• Plan fault-tolerant systems")

if __name__ == "__main__":
    test_articulation_analysis()
    demonstrate_tarjan_algorithm()
    demonstrate_network_applications()
    analyze_connectivity_concepts()
    demonstrate_practical_applications()

"""
Articulation Points and Bridges Concepts:
1. Tarjan's Algorithm for Cut Vertex and Cut Edge Detection
2. Network Connectivity and Robustness Analysis
3. Biconnected and Bridge-Connected Component Analysis
4. Graph Vulnerability Assessment and Fault Tolerance
5. Critical Infrastructure Identification and Protection

Key Problem Insights:
- Articulation points are critical vertices for connectivity
- Bridges are critical edges for network communication
- Tarjan's algorithm finds both in single O(V + E) traversal
- Essential for network reliability and robustness analysis

Algorithm Strategy:
1. DFS traversal with discovery and low-link times
2. Articulation point conditions: root with multiple children, non-root with cutoff
3. Bridge condition: no back edge from subtree to ancestors
4. Single algorithm finds both articulation points and bridges

Network Analysis Applications:
- Infrastructure vulnerability assessment
- Social network influence analysis
- Computer network reliability planning
- Biological network critical component identification
- Circuit design fault tolerance optimization

Connectivity Properties:
- Biconnected components have no articulation points
- Bridge-connected components have no bridges
- Network robustness measured by critical element density
- Fault tolerance requires redundant connectivity paths

Real-world Impact:
- Network infrastructure investment prioritization
- Fault-tolerant system design
- Emergency response planning
- Security vulnerability assessment
- Resource allocation optimization

This implementation provides complete articulation analysis
essential for network reliability and infrastructure protection.
"""
