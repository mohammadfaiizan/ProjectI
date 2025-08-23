"""
Network Flow Fundamentals - Comprehensive Theory and Implementation
Difficulty: Easy

This file provides fundamental concepts, algorithms, and applications of network flow theory.
Covers basic flow networks, flow properties, and essential algorithms for understanding
more advanced network flow problems.

Key Concepts:
1. Flow Network Definition and Properties
2. Flow Conservation and Capacity Constraints
3. Residual Networks and Augmenting Paths
4. Cut Concepts and Min-Cut Max-Flow Theorem
5. Basic Flow Algorithms Implementation
6. Flow Network Applications and Modeling
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass

@dataclass
class FlowEdge:
    """Represents an edge in a flow network"""
    from_vertex: int
    to_vertex: int
    capacity: int
    flow: int = 0
    
    def residual_capacity(self) -> int:
        """Get remaining capacity on this edge"""
        return self.capacity - self.flow
    
    def reverse_capacity(self) -> int:
        """Get capacity of reverse edge (current flow)"""
        return self.flow

class NetworkFlowFundamentals:
    """Comprehensive implementation of network flow fundamentals"""
    
    def __init__(self):
        self.reset_network()
    
    def reset_network(self):
        """Reset network state"""
        self.vertices = set()
        self.edges = []
        self.adjacency = defaultdict(list)
        self.edge_map = {}
        self.statistics = {
            'vertices': 0,
            'edges': 0,
            'total_capacity': 0,
            'flow_value': 0,
            'iterations': 0
        }
    
    def add_edge(self, u: int, v: int, capacity: int) -> None:
        """
        Approach 1: Basic Edge Addition
        
        Add edge to flow network with capacity constraint.
        
        Time: O(1)
        Space: O(1)
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        edge = FlowEdge(u, v, capacity)
        self.edges.append(edge)
        self.adjacency[u].append(len(self.edges) - 1)
        self.edge_map[(u, v)] = len(self.edges) - 1
        
        # Add reverse edge with 0 capacity
        reverse_edge = FlowEdge(v, u, 0)
        self.edges.append(reverse_edge)
        self.adjacency[v].append(len(self.edges) - 1)
        self.edge_map[(v, u)] = len(self.edges) - 1
        
        self.vertices.add(u)
        self.vertices.add(v)
        self.statistics['vertices'] = len(self.vertices)
        self.statistics['edges'] = len(self.edges) // 2
        self.statistics['total_capacity'] += capacity
    
    def build_from_adjacency_matrix(self, matrix: List[List[int]]) -> None:
        """
        Approach 2: Build Network from Adjacency Matrix
        
        Construct flow network from capacity matrix.
        
        Time: O(V^2)
        Space: O(V^2)
        """
        self.reset_network()
        n = len(matrix)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0:
                    self.add_edge(i, j, matrix[i][j])
    
    def build_from_edge_list(self, edges: List[Tuple[int, int, int]]) -> None:
        """
        Approach 3: Build Network from Edge List
        
        Construct flow network from list of (u, v, capacity) tuples.
        
        Time: O(E)
        Space: O(E)
        """
        self.reset_network()
        
        for u, v, capacity in edges:
            self.add_edge(u, v, capacity)
    
    def get_residual_capacity(self, u: int, v: int) -> int:
        """
        Approach 4: Residual Capacity Calculation
        
        Get residual capacity between two vertices.
        
        Time: O(1)
        Space: O(1)
        """
        if (u, v) not in self.edge_map:
            return 0
        
        edge_idx = self.edge_map[(u, v)]
        return self.edges[edge_idx].residual_capacity()
    
    def find_augmenting_path_bfs(self, source: int, sink: int) -> Optional[List[int]]:
        """
        Approach 5: BFS Augmenting Path Finding
        
        Find shortest augmenting path using BFS (Edmonds-Karp style).
        
        Time: O(V + E)
        Space: O(V)
        """
        if source == sink:
            return None
        
        parent = {source: None}
        visited = {source}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            
            # Check all outgoing edges
            for edge_idx in self.adjacency[current]:
                edge = self.edges[edge_idx]
                neighbor = edge.to_vertex
                
                if neighbor not in visited and edge.residual_capacity() > 0:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
                    
                    if neighbor == sink:
                        # Reconstruct path
                        path = []
                        current_node = sink
                        while current_node is not None:
                            path.append(current_node)
                            current_node = parent[current_node]
                        return path[::-1]
        
        return None
    
    def find_augmenting_path_dfs(self, source: int, sink: int) -> Optional[List[int]]:
        """
        Approach 6: DFS Augmenting Path Finding
        
        Find augmenting path using DFS (Ford-Fulkerson style).
        
        Time: O(V + E)
        Space: O(V)
        """
        visited = set()
        path = []
        
        def dfs(current):
            if current == sink:
                return True
            
            visited.add(current)
            path.append(current)
            
            for edge_idx in self.adjacency[current]:
                edge = self.edges[edge_idx]
                neighbor = edge.to_vertex
                
                if neighbor not in visited and edge.residual_capacity() > 0:
                    if dfs(neighbor):
                        return True
            
            path.pop()
            return False
        
        if dfs(source):
            return path
        return None
    
    def get_bottleneck_capacity(self, path: List[int]) -> int:
        """
        Approach 7: Bottleneck Capacity Calculation
        
        Find minimum residual capacity along a path.
        
        Time: O(path_length)
        Space: O(1)
        """
        if len(path) < 2:
            return 0
        
        min_capacity = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            capacity = self.get_residual_capacity(u, v)
            min_capacity = min(min_capacity, capacity)
        
        return min_capacity if min_capacity != float('inf') else 0
    
    def push_flow_along_path(self, path: List[int], flow_amount: int) -> None:
        """
        Approach 8: Flow Pushing Along Path
        
        Push flow along augmenting path and update residual capacities.
        
        Time: O(path_length)
        Space: O(1)
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Forward edge
            if (u, v) in self.edge_map:
                edge_idx = self.edge_map[(u, v)]
                self.edges[edge_idx].flow += flow_amount
            
            # Reverse edge
            if (v, u) in self.edge_map:
                reverse_idx = self.edge_map[(v, u)]
                self.edges[reverse_idx].flow -= flow_amount
    
    def compute_max_flow_edmonds_karp(self, source: int, sink: int) -> Dict:
        """
        Approach 9: Edmonds-Karp Maximum Flow
        
        Compute maximum flow using BFS-based Ford-Fulkerson.
        
        Time: O(V * E^2)
        Space: O(V + E)
        """
        max_flow = 0
        iterations = 0
        paths_found = []
        
        while True:
            iterations += 1
            
            # Find augmenting path
            path = self.find_augmenting_path_bfs(source, sink)
            
            if path is None:
                break
            
            # Calculate bottleneck capacity
            bottleneck = self.get_bottleneck_capacity(path)
            
            if bottleneck == 0:
                break
            
            # Push flow
            self.push_flow_along_path(path, bottleneck)
            max_flow += bottleneck
            paths_found.append((path[:], bottleneck))
        
        self.statistics['flow_value'] = max_flow
        self.statistics['iterations'] = iterations
        
        return {
            'max_flow': max_flow,
            'iterations': iterations,
            'paths_found': paths_found,
            'algorithm': 'edmonds_karp'
        }
    
    def find_min_cut(self, source: int, sink: int) -> Dict:
        """
        Approach 10: Min-Cut Finding
        
        Find minimum cut after computing maximum flow.
        
        Time: O(V + E)
        Space: O(V)
        """
        # First compute max flow
        flow_result = self.compute_max_flow_edmonds_karp(source, sink)
        
        # Find reachable vertices from source in residual graph
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        
        while queue:
            current = queue.popleft()
            
            for edge_idx in self.adjacency[current]:
                edge = self.edges[edge_idx]
                neighbor = edge.to_vertex
                
                if neighbor not in reachable and edge.residual_capacity() > 0:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        # Find cut edges
        cut_edges = []
        cut_capacity = 0
        
        for edge in self.edges:
            if (edge.from_vertex in reachable and 
                edge.to_vertex not in reachable and 
                edge.capacity > 0):
                cut_edges.append((edge.from_vertex, edge.to_vertex, edge.capacity))
                cut_capacity += edge.capacity
        
        return {
            'cut_capacity': cut_capacity,
            'cut_edges': cut_edges,
            'source_side': list(reachable),
            'sink_side': [v for v in self.vertices if v not in reachable],
            'max_flow': flow_result['max_flow']
        }
    
    def verify_flow_properties(self, source: int, sink: int) -> Dict:
        """
        Approach 11: Flow Properties Verification
        
        Verify that current flow satisfies all flow network properties.
        
        Time: O(V + E)
        Space: O(V)
        """
        violations = []
        
        # Check capacity constraints
        for edge in self.edges:
            if edge.flow > edge.capacity:
                violations.append(f"Capacity violation: edge ({edge.from_vertex}, {edge.to_vertex})")
            if edge.flow < 0:
                violations.append(f"Negative flow: edge ({edge.from_vertex}, {edge.to_vertex})")
        
        # Check flow conservation
        for vertex in self.vertices:
            if vertex == source or vertex == sink:
                continue
            
            inflow = sum(edge.flow for edge in self.edges if edge.to_vertex == vertex)
            outflow = sum(edge.flow for edge in self.edges if edge.from_vertex == vertex)
            
            if abs(inflow - outflow) > 1e-9:
                violations.append(f"Flow conservation violation at vertex {vertex}")
        
        # Calculate net flow out of source
        source_outflow = sum(edge.flow for edge in self.edges if edge.from_vertex == source)
        source_inflow = sum(edge.flow for edge in self.edges if edge.to_vertex == source)
        net_source_flow = source_outflow - source_inflow
        
        # Calculate net flow into sink
        sink_inflow = sum(edge.flow for edge in self.edges if edge.to_vertex == sink)
        sink_outflow = sum(edge.flow for edge in self.edges if edge.from_vertex == sink)
        net_sink_flow = sink_inflow - sink_outflow
        
        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'source_flow': net_source_flow,
            'sink_flow': net_sink_flow,
            'flow_balance': abs(net_source_flow - net_sink_flow) < 1e-9
        }
    
    def analyze_network_structure(self) -> Dict:
        """
        Approach 12: Network Structure Analysis
        
        Analyze structural properties of the flow network.
        
        Time: O(V + E)
        Space: O(V)
        """
        if not self.vertices:
            return {'empty_network': True}
        
        # Basic statistics
        num_vertices = len(self.vertices)
        num_edges = len([e for e in self.edges if e.capacity > 0])
        total_capacity = sum(e.capacity for e in self.edges if e.capacity > 0)
        
        # Degree analysis
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for edge in self.edges:
            if edge.capacity > 0:
                out_degree[edge.from_vertex] += 1
                in_degree[edge.to_vertex] += 1
        
        # Find sources and sinks
        sources = [v for v in self.vertices if in_degree[v] == 0 and out_degree[v] > 0]
        sinks = [v for v in self.vertices if out_degree[v] == 0 and in_degree[v] > 0]
        
        # Connectivity analysis
        def is_connected():
            if num_vertices <= 1:
                return True
            
            visited = set()
            queue = deque([next(iter(self.vertices))])
            visited.add(queue[0])
            
            while queue:
                current = queue.popleft()
                for edge_idx in self.adjacency[current]:
                    edge = self.edges[edge_idx]
                    neighbor = edge.to_vertex
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == num_vertices
        
        return {
            'vertices': num_vertices,
            'edges': num_edges,
            'total_capacity': total_capacity,
            'average_degree': (2 * num_edges) / num_vertices if num_vertices > 0 else 0,
            'sources': sources,
            'sinks': sinks,
            'is_connected': is_connected(),
            'max_in_degree': max(in_degree.values()) if in_degree else 0,
            'max_out_degree': max(out_degree.values()) if out_degree else 0
        }

def test_network_flow_fundamentals():
    """Test network flow fundamental operations"""
    print("=== Testing Network Flow Fundamentals ===")
    
    # Test cases
    test_networks = [
        # Simple network
        ([(0, 1, 10), (0, 2, 8), (1, 3, 5), (1, 2, 2), (2, 3, 10)], 0, 3, 13, "Simple network"),
        
        # Linear chain
        ([(0, 1, 5), (1, 2, 3), (2, 3, 7)], 0, 3, 3, "Linear chain"),
        
        # Parallel paths
        ([(0, 1, 10), (0, 2, 10), (1, 3, 10), (2, 3, 10)], 0, 3, 20, "Parallel paths"),
        
        # Complex network
        ([(0, 1, 16), (0, 2, 13), (1, 2, 10), (1, 3, 12), (2, 1, 4), (2, 4, 14), 
          (3, 2, 9), (3, 5, 20), (4, 3, 7), (4, 5, 4)], 0, 5, 23, "Complex network"),
    ]
    
    for edges, source, sink, expected_flow, description in test_networks:
        print(f"\n--- {description} ---")
        
        # Create network
        network = NetworkFlowFundamentals()
        network.build_from_edge_list(edges)
        
        # Analyze structure
        structure = network.analyze_network_structure()
        print(f"Vertices: {structure['vertices']}, Edges: {structure['edges']}")
        print(f"Total capacity: {structure['total_capacity']}")
        
        # Compute max flow
        flow_result = network.compute_max_flow_edmonds_karp(source, sink)
        print(f"Max flow: {flow_result['max_flow']} (expected: {expected_flow})")
        print(f"Iterations: {flow_result['iterations']}")
        
        # Find min cut
        cut_result = network.find_min_cut(source, sink)
        print(f"Min cut capacity: {cut_result['cut_capacity']}")
        print(f"Cut edges: {cut_result['cut_edges']}")
        
        # Verify flow properties
        verification = network.verify_flow_properties(source, sink)
        print(f"Flow valid: {verification['valid']}")
        if verification['violations']:
            print(f"Violations: {verification['violations']}")

def demonstrate_flow_concepts():
    """Demonstrate fundamental flow network concepts"""
    print("\n=== Flow Network Concepts Demo ===")
    
    print("1. **Flow Network Definition:**")
    print("   • Directed graph G = (V, E)")
    print("   • Capacity function c: E → ℝ⁺")
    print("   • Source vertex s (no incoming edges)")
    print("   • Sink vertex t (no outgoing edges)")
    
    print("\n2. **Flow Function Properties:**")
    print("   • Capacity constraint: 0 ≤ f(u,v) ≤ c(u,v)")
    print("   • Flow conservation: Σf(u,v) = Σf(v,w) for v ≠ s,t")
    print("   • Flow value: |f| = Σf(s,v) - Σf(v,s)")
    
    print("\n3. **Residual Network:**")
    print("   • Residual capacity: cf(u,v) = c(u,v) - f(u,v)")
    print("   • Reverse edge capacity: cf(v,u) = f(u,v)")
    print("   • Augmenting path: s-t path in residual network")
    
    print("\n4. **Cut Concepts:**")
    print("   • s-t cut: partition (S,T) with s ∈ S, t ∈ T")
    print("   • Cut capacity: Σc(u,v) for u ∈ S, v ∈ T")
    print("   • Min-cut: cut with minimum capacity")
    print("   • Max-flow min-cut theorem: max flow = min cut")
    
    # Example demonstration
    network = NetworkFlowFundamentals()
    edges = [(0, 1, 10), (0, 2, 8), (1, 3, 5), (1, 2, 2), (2, 3, 10)]
    network.build_from_edge_list(edges)
    
    print(f"\nExample Network: {edges}")
    
    # Show augmenting path finding
    path = network.find_augmenting_path_bfs(0, 3)
    print(f"First augmenting path: {path}")
    
    if path:
        bottleneck = network.get_bottleneck_capacity(path)
        print(f"Bottleneck capacity: {bottleneck}")

def analyze_algorithm_theory():
    """Analyze theoretical aspects of flow algorithms"""
    print("\n=== Flow Algorithm Theory ===")
    
    print("Algorithm Complexity Analysis:")
    
    print("\n1. **Ford-Fulkerson Method:**")
    print("   • Generic approach using augmenting paths")
    print("   • Time: O(E * |f*|) where |f*| is max flow value")
    print("   • Can be exponential with poor path selection")
    print("   • Correctness based on augmenting path theorem")
    
    print("\n2. **Edmonds-Karp Algorithm:**")
    print("   • Ford-Fulkerson with BFS path selection")
    print("   • Time: O(V * E²) - polynomial guarantee")
    print("   • Shortest paths ensure efficient convergence")
    print("   • Each edge becomes critical at most V/2 times")
    
    print("\n3. **Path Selection Strategies:**")
    print("   • BFS: shortest augmenting paths (Edmonds-Karp)")
    print("   • DFS: depth-first paths (basic Ford-Fulkerson)")
    print("   • Capacity scaling: high-capacity paths first")
    print("   • Shortest path: minimum number of edges")
    
    print("\n4. **Optimality Conditions:**")
    print("   • No augmenting paths exist")
    print("   • Flow value equals min-cut capacity")
    print("   • Residual graph has no s-t path")
    print("   • Complementary slackness conditions")
    
    print("\n5. **Advanced Algorithms:**")
    print("   • Push-relabel: O(V²E) or O(V³)")
    print("   • Dinic's algorithm: O(V²E)")
    print("   • ISAP (Improved Shortest Augmenting Path)")
    print("   • Capacity scaling: O(E² log(max_capacity))")

def demonstrate_applications():
    """Demonstrate practical applications of network flow"""
    print("\n=== Network Flow Applications ===")
    
    print("Real-World Applications:")
    
    print("\n1. **Transportation Networks:**")
    print("   • Traffic flow optimization")
    print("   • Railway capacity planning")
    print("   • Supply chain logistics")
    print("   • Pipeline flow management")
    
    print("\n2. **Communication Networks:**")
    print("   • Internet routing protocols")
    print("   • Bandwidth allocation")
    print("   • Network reliability analysis")
    print("   • Data center traffic engineering")
    
    print("\n3. **Resource Allocation:**")
    print("   • Job assignment problems")
    print("   • Project scheduling")
    print("   • Facility location")
    print("   • Resource distribution")
    
    print("\n4. **Bipartite Matching:**")
    print("   • Marriage problem")
    print("   • Task assignment")
    print("   • Course scheduling")
    print("   • Organ donation matching")
    
    print("\n5. **Image Processing:**")
    print("   • Image segmentation")
    print("   • Object detection")
    print("   • Stereo vision")
    print("   • Medical image analysis")
    
    print("\nModeling Techniques:")
    print("• Node splitting for vertex capacities")
    print("• Multiple sources/sinks via super nodes")
    print("• Lower bounds via flow transformation")
    print("• Cost considerations in min-cost flow")

if __name__ == "__main__":
    test_network_flow_fundamentals()
    demonstrate_flow_concepts()
    analyze_algorithm_theory()
    demonstrate_applications()

"""
Network Flow Fundamentals - Key Insights:

1. **Core Concepts:**
   - Flow networks model resource transportation
   - Capacity constraints and flow conservation
   - Residual networks capture remaining potential
   - Augmenting paths enable flow improvement

2. **Fundamental Algorithms:**
   - Ford-Fulkerson method provides framework
   - Edmonds-Karp guarantees polynomial time
   - Path selection strategy affects performance
   - Min-cut max-flow theorem ensures optimality

3. **Implementation Considerations:**
   - Residual graph representation
   - Efficient path finding algorithms
   - Flow pushing and capacity updates
   - Numerical stability and precision

4. **Theoretical Foundations:**
   - Max-flow min-cut duality
   - Augmenting path characterization
   - Polynomial-time complexity analysis
   - Optimality conditions and certificates

5. **Practical Applications:**
   - Transportation and logistics
   - Network design and optimization
   - Resource allocation problems
   - Matching and assignment tasks

Network flow theory provides fundamental tools
for optimization in networked systems.
"""
