"""
Multi-Source Multi-Sink Flow - Comprehensive Implementation
Difficulty: Medium

This file provides implementations for network flow problems with multiple sources
and/or multiple sinks, including transformations to standard max-flow problems
and specialized algorithms for multi-terminal flow scenarios.

Key Concepts:
1. Super Source and Super Sink Transformations
2. Multi-Terminal Flow Problems
3. Steiner Tree Flow Applications
4. Concurrent Flow Problems
5. Multi-Commodity Flow Basics
6. Flow Decomposition Techniques
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass

@dataclass
class MultiFlowEdge:
    """Edge for multi-source multi-sink flow networks"""
    to: int
    capacity: int
    flow: int = 0
    cost: int = 0
    commodity: int = -1  # For multi-commodity flows
    
    def residual_capacity(self) -> int:
        return self.capacity - self.flow

class MultiSourceMultiSinkFlow:
    """Multi-source multi-sink flow algorithm implementations"""
    
    def __init__(self):
        self.reset_network()
    
    def reset_network(self):
        """Reset network state"""
        self.graph = defaultdict(list)
        self.vertices = set()
        self.statistics = {
            'transformations': 0,
            'flow_computations': 0,
            'commodities_processed': 0,
            'iterations': 0
        }
    
    def add_edge(self, u: int, v: int, capacity: int, cost: int = 0, commodity: int = -1):
        """Add edge with reverse edge for residual graph"""
        # Forward edge
        forward_edge = MultiFlowEdge(v, capacity, 0, cost, commodity)
        self.graph[u].append(forward_edge)
        
        # Reverse edge
        reverse_edge = MultiFlowEdge(u, 0, 0, -cost, commodity)
        self.graph[v].append(reverse_edge)
        
        self.vertices.add(u)
        self.vertices.add(v)
    
    def max_flow_super_source_sink(self, sources: List[int], sinks: List[int], 
                                  source_capacities: List[int] = None,
                                  sink_capacities: List[int] = None) -> Dict:
        """
        Approach 1: Super Source and Super Sink Transformation
        
        Transform multi-source multi-sink problem to single source-sink.
        
        Time: O(max_flow_time)
        Space: O(V + E)
        """
        self.statistics['transformations'] += 1
        
        # Create super source and super sink
        super_source = max(self.vertices) + 1
        super_sink = super_source + 1
        
        # Default capacities if not provided
        if source_capacities is None:
            source_capacities = [float('inf')] * len(sources)
        if sink_capacities is None:
            sink_capacities = [float('inf')] * len(sinks)
        
        # Add edges from super source to all sources
        for i, source in enumerate(sources):
            self.add_edge(super_source, source, source_capacities[i])
        
        # Add edges from all sinks to super sink
        for i, sink in enumerate(sinks):
            self.add_edge(sink, super_sink, sink_capacities[i])
        
        # Solve single-source single-sink max-flow
        result = self._edmonds_karp_max_flow(super_source, super_sink)
        
        # Extract individual source-sink flows
        source_flows = {}
        sink_flows = {}
        
        for i, source in enumerate(sources):
            total_flow = 0
            for edge in self.graph[super_source]:
                if edge.to == source:
                    total_flow = edge.flow
                    break
            source_flows[source] = total_flow
        
        for i, sink in enumerate(sinks):
            total_flow = 0
            for edge in self.graph[sink]:
                if edge.to == super_sink:
                    total_flow = edge.flow
                    break
            sink_flows[sink] = total_flow
        
        return {
            'max_flow': result['max_flow'],
            'source_flows': source_flows,
            'sink_flows': sink_flows,
            'algorithm': 'super_source_sink',
            'statistics': self.statistics.copy()
        }
    
    def multi_terminal_max_flow(self, terminals: List[int], 
                               terminal_demands: List[int]) -> Dict:
        """
        Approach 2: Multi-Terminal Max Flow
        
        Find maximum flow satisfying terminal demands.
        
        Time: O(k * max_flow_time) where k is number of terminals
        Space: O(V + E)
        """
        if len(terminals) != len(terminal_demands):
            raise ValueError("Terminals and demands must have same length")
        
        # Separate sources (positive demand) and sinks (negative demand)
        sources = []
        sinks = []
        source_supplies = []
        sink_demands = []
        
        for terminal, demand in zip(terminals, terminal_demands):
            if demand > 0:
                sources.append(terminal)
                source_supplies.append(demand)
            elif demand < 0:
                sinks.append(terminal)
                sink_demands.append(-demand)
        
        if not sources or not sinks:
            return {'feasible': False, 'reason': 'No sources or no sinks'}
        
        # Check supply-demand balance
        total_supply = sum(source_supplies)
        total_demand = sum(sink_demands)
        
        if total_supply != total_demand:
            return {'feasible': False, 'reason': 'Supply-demand imbalance'}
        
        # Use super source-sink transformation
        result = self.max_flow_super_source_sink(sources, sinks, 
                                               source_supplies, sink_demands)
        
        # Check if all demands are satisfied
        feasible = result['max_flow'] == total_supply
        
        return {
            'feasible': feasible,
            'max_flow': result['max_flow'],
            'total_supply': total_supply,
            'total_demand': total_demand,
            'source_flows': result['source_flows'],
            'sink_flows': result['sink_flows'],
            'algorithm': 'multi_terminal'
        }
    
    def concurrent_flow_problem(self, flow_requests: List[Tuple[int, int, int]]) -> Dict:
        """
        Approach 3: Concurrent Flow Problem
        
        Maximize concurrent satisfaction of multiple flow requests.
        
        Time: O(binary_search_iterations * max_flow_time)
        Space: O(V + E)
        """
        # flow_requests: [(source, sink, demand), ...]
        
        def can_satisfy_fraction(fraction: float) -> bool:
            """Check if fraction of all demands can be satisfied concurrently"""
            # Create temporary network with scaled demands
            temp_sources = []
            temp_sinks = []
            temp_source_caps = []
            temp_sink_caps = []
            
            for source, sink, demand in flow_requests:
                scaled_demand = int(demand * fraction)
                if scaled_demand > 0:
                    temp_sources.append(source)
                    temp_sinks.append(sink)
                    temp_source_caps.append(scaled_demand)
                    temp_sink_caps.append(scaled_demand)
            
            if not temp_sources:
                return True
            
            # Test feasibility
            result = self.max_flow_super_source_sink(temp_sources, temp_sinks,
                                                   temp_source_caps, temp_sink_caps)
            
            expected_flow = sum(temp_source_caps)
            return result['max_flow'] >= expected_flow
        
        # Binary search for maximum feasible fraction
        left, right = 0.0, 1.0
        epsilon = 1e-6
        
        while right - left > epsilon:
            mid = (left + right) / 2
            if can_satisfy_fraction(mid):
                left = mid
            else:
                right = mid
        
        max_fraction = left
        
        # Compute actual flows for maximum fraction
        final_result = {}
        if max_fraction > 0:
            sources = [req[0] for req in flow_requests]
            sinks = [req[1] for req in flow_requests]
            source_caps = [int(req[2] * max_fraction) for req in flow_requests]
            sink_caps = source_caps[:]
            
            flow_result = self.max_flow_super_source_sink(sources, sinks,
                                                        source_caps, sink_caps)
            final_result = flow_result
        
        return {
            'max_concurrent_fraction': max_fraction,
            'flow_result': final_result,
            'algorithm': 'concurrent_flow'
        }
    
    def multi_commodity_flow_basic(self, commodities: List[Tuple[int, int, int]]) -> Dict:
        """
        Approach 4: Basic Multi-Commodity Flow
        
        Route multiple commodities through shared network.
        
        Time: O(k * max_flow_time) where k is number of commodities
        Space: O(k * (V + E))
        """
        # commodities: [(source, sink, demand), ...]
        
        commodity_flows = {}
        total_flow = 0
        
        # Process each commodity separately (greedy approach)
        for i, (source, sink, demand) in enumerate(commodities):
            self.statistics['commodities_processed'] += 1
            
            # Find maximum flow for this commodity
            result = self._edmonds_karp_max_flow(source, sink)
            
            commodity_flow = min(result['max_flow'], demand)
            commodity_flows[i] = {
                'source': source,
                'sink': sink,
                'demand': demand,
                'flow': commodity_flow,
                'satisfaction': commodity_flow / demand if demand > 0 else 1.0
            }
            
            total_flow += commodity_flow
            
            # Update residual capacities (simplified - doesn't handle conflicts properly)
            if commodity_flow > 0:
                self._update_residual_capacities(source, sink, commodity_flow)
        
        return {
            'total_flow': total_flow,
            'commodity_flows': commodity_flows,
            'average_satisfaction': sum(cf['satisfaction'] for cf in commodity_flows.values()) / len(commodities),
            'algorithm': 'multi_commodity_basic'
        }
    
    def steiner_tree_flow(self, root: int, terminals: List[int], 
                         terminal_demands: List[int]) -> Dict:
        """
        Approach 5: Steiner Tree Flow Problem
        
        Find minimum cost flow connecting root to all terminals.
        
        Time: O(Steiner_tree_time + max_flow_time)
        Space: O(V + E)
        """
        # Simplified Steiner tree approximation using shortest paths
        steiner_vertices = set([root] + terminals)
        
        # Find shortest paths from root to all terminals
        distances, paths = self._all_shortest_paths(root)
        
        # Add intermediate vertices on shortest paths to Steiner set
        for terminal in terminals:
            if terminal in paths:
                path = paths[terminal]
                steiner_vertices.update(path)
        
        # Create subgraph with only Steiner vertices
        steiner_edges = []
        for u in steiner_vertices:
            for edge in self.graph[u]:
                if edge.to in steiner_vertices and edge.capacity > 0:
                    steiner_edges.append((u, edge.to, edge.capacity, edge.cost))
        
        # Solve flow problem on Steiner subgraph
        steiner_sources = [root]
        steiner_sinks = terminals
        steiner_source_caps = [sum(terminal_demands)]
        steiner_sink_caps = terminal_demands
        
        result = self.max_flow_super_source_sink(steiner_sources, steiner_sinks,
                                               steiner_source_caps, steiner_sink_caps)
        
        return {
            'steiner_vertices': list(steiner_vertices),
            'steiner_edges': steiner_edges,
            'flow_result': result,
            'algorithm': 'steiner_tree_flow'
        }
    
    def flow_decomposition(self, source: int, sink: int) -> Dict:
        """
        Approach 6: Flow Decomposition
        
        Decompose flow into paths and cycles.
        
        Time: O(flow_value * (V + E))
        Space: O(V + E)
        """
        # First find maximum flow
        flow_result = self._edmonds_karp_max_flow(source, sink)
        max_flow = flow_result['max_flow']
        
        if max_flow == 0:
            return {'paths': [], 'cycles': [], 'total_flow': 0}
        
        paths = []
        cycles = []
        remaining_flow = max_flow
        
        while remaining_flow > 0:
            # Find a path from source to sink with positive flow
            path, path_flow = self._find_flow_path(source, sink)
            
            if path and path_flow > 0:
                paths.append((path, path_flow))
                remaining_flow -= path_flow
                
                # Reduce flow along this path
                self._reduce_flow_along_path(path, path_flow)
            else:
                # Look for cycles with positive flow
                cycle, cycle_flow = self._find_flow_cycle()
                
                if cycle and cycle_flow > 0:
                    cycles.append((cycle, cycle_flow))
                    self._reduce_flow_along_path(cycle, cycle_flow)
                else:
                    break  # No more paths or cycles
        
        return {
            'paths': paths,
            'cycles': cycles,
            'total_flow': max_flow,
            'decomposed_flow': sum(pf for _, pf in paths),
            'algorithm': 'flow_decomposition'
        }
    
    def _edmonds_karp_max_flow(self, source: int, sink: int) -> Dict:
        """Basic Edmonds-Karp implementation for internal use"""
        max_flow = 0
        iterations = 0
        
        while True:
            iterations += 1
            self.statistics['iterations'] += 1
            
            # BFS to find augmenting path
            parent = {}
            queue = deque([source])
            parent[source] = None
            
            while queue and sink not in parent:
                u = queue.popleft()
                
                for edge in self.graph[u]:
                    v = edge.to
                    if v not in parent and edge.residual_capacity() > 0:
                        parent[v] = (u, edge)
                        queue.append(v)
            
            if sink not in parent:
                break
            
            # Find bottleneck capacity
            bottleneck = float('inf')
            current = sink
            
            while current != source:
                u, edge = parent[current]
                bottleneck = min(bottleneck, edge.residual_capacity())
                current = u
            
            # Push flow
            current = sink
            while current != source:
                u, edge = parent[current]
                edge.flow += bottleneck
                # Find reverse edge and update
                for rev_edge in self.graph[current]:
                    if rev_edge.to == u:
                        rev_edge.flow -= bottleneck
                        break
                current = u
            
            max_flow += bottleneck
        
        self.statistics['flow_computations'] += 1
        
        return {
            'max_flow': max_flow,
            'iterations': iterations
        }
    
    def _all_shortest_paths(self, source: int) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Find shortest paths from source to all vertices"""
        distances = {source: 0}
        paths = {source: [source]}
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            
            for edge in self.graph[u]:
                v = edge.to
                if v not in distances and edge.capacity > 0:
                    distances[v] = distances[u] + 1
                    paths[v] = paths[u] + [v]
                    queue.append(v)
        
        return distances, paths
    
    def _find_flow_path(self, source: int, sink: int) -> Tuple[List[int], int]:
        """Find path with positive flow from source to sink"""
        parent = {}
        queue = deque([source])
        parent[source] = None
        
        while queue and sink not in parent:
            u = queue.popleft()
            
            for edge in self.graph[u]:
                v = edge.to
                if v not in parent and edge.flow > 0:
                    parent[v] = (u, edge)
                    queue.append(v)
        
        if sink not in parent:
            return [], 0
        
        # Reconstruct path and find minimum flow
        path = []
        min_flow = float('inf')
        current = sink
        
        while current != source:
            path.append(current)
            u, edge = parent[current]
            min_flow = min(min_flow, edge.flow)
            current = u
        
        path.append(source)
        path.reverse()
        
        return path, min_flow
    
    def _find_flow_cycle(self) -> Tuple[List[int], int]:
        """Find cycle with positive flow"""
        visited = set()
        
        for start in self.vertices:
            if start in visited:
                continue
            
            path = []
            current = start
            path_set = set()
            
            while current not in visited:
                if current in path_set:
                    # Found cycle
                    cycle_start_idx = path.index(current)
                    cycle = path[cycle_start_idx:]
                    
                    # Find minimum flow in cycle
                    min_flow = float('inf')
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        
                        for edge in self.graph[u]:
                            if edge.to == v and edge.flow > 0:
                                min_flow = min(min_flow, edge.flow)
                                break
                    
                    if min_flow > 0 and min_flow != float('inf'):
                        return cycle, min_flow
                    break
                
                path.append(current)
                path_set.add(current)
                
                # Find next vertex with positive flow
                next_vertex = None
                for edge in self.graph[current]:
                    if edge.flow > 0:
                        next_vertex = edge.to
                        break
                
                if next_vertex is None:
                    break
                
                current = next_vertex
            
            visited.update(path_set)
        
        return [], 0
    
    def _reduce_flow_along_path(self, path: List[int], flow_amount: int):
        """Reduce flow along given path"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            for edge in self.graph[u]:
                if edge.to == v:
                    edge.flow -= flow_amount
                    break
            
            for edge in self.graph[v]:
                if edge.to == u:
                    edge.flow += flow_amount
                    break
    
    def _update_residual_capacities(self, source: int, sink: int, flow_amount: int):
        """Update residual capacities after routing flow (simplified)"""
        # This is a simplified version - proper multi-commodity flow
        # requires more sophisticated capacity management
        pass

def test_multi_source_multi_sink_flow():
    """Test multi-source multi-sink flow algorithms"""
    print("=== Testing Multi-Source Multi-Sink Flow ===")
    
    # Create test network
    flow_solver = MultiSourceMultiSinkFlow()
    
    # Build test graph
    edges = [
        (0, 2, 10), (1, 2, 8), (2, 4, 12),
        (2, 5, 15), (4, 6, 10), (5, 6, 8),
        (4, 7, 5), (5, 7, 7)
    ]
    
    for u, v, capacity in edges:
        flow_solver.add_edge(u, v, capacity)
    
    print(f"Test network: {len(flow_solver.vertices)} vertices, {len(edges)} edges")
    
    # Test 1: Multi-source single-sink
    sources = [0, 1]
    sinks = [6, 7]
    
    print(f"\nTest 1: Multi-source multi-sink")
    print(f"Sources: {sources}, Sinks: {sinks}")
    
    result = flow_solver.max_flow_super_source_sink(sources, sinks)
    print(f"Max flow: {result['max_flow']}")
    print(f"Source flows: {result['source_flows']}")
    print(f"Sink flows: {result['sink_flows']}")
    
    # Test 2: Multi-terminal flow
    print(f"\nTest 2: Multi-terminal flow")
    terminals = [0, 1, 6, 7]
    demands = [10, 8, -12, -6]  # Positive = supply, Negative = demand
    
    # Reset flows
    for u in flow_solver.graph:
        for edge in flow_solver.graph[u]:
            edge.flow = 0
    
    terminal_result = flow_solver.multi_terminal_max_flow(terminals, demands)
    print(f"Feasible: {terminal_result['feasible']}")
    if terminal_result['feasible']:
        print(f"Max flow: {terminal_result['max_flow']}")
    
    # Test 3: Flow decomposition
    print(f"\nTest 3: Flow decomposition")
    
    # Reset and find flow first
    for u in flow_solver.graph:
        for edge in flow_solver.graph[u]:
            edge.flow = 0
    
    decomp_result = flow_solver.flow_decomposition(0, 7)
    print(f"Total flow: {decomp_result['total_flow']}")
    print(f"Number of paths: {len(decomp_result['paths'])}")
    print(f"Number of cycles: {len(decomp_result['cycles'])}")

def demonstrate_multi_flow_applications():
    """Demonstrate applications of multi-source multi-sink flows"""
    print("\n=== Multi-Flow Applications Demo ===")
    
    print("Real-World Applications:")
    
    print("\n1. **Supply Chain Management:**")
    print("   • Multiple suppliers to multiple customers")
    print("   • Warehouse distribution networks")
    print("   • Manufacturing resource allocation")
    print("   • Inventory balancing across locations")
    
    print("\n2. **Transportation Networks:**")
    print("   • Multi-origin multi-destination routing")
    print("   • Public transit system optimization")
    print("   • Freight distribution planning")
    print("   • Emergency evacuation routing")
    
    print("\n3. **Communication Networks:**")
    print("   • Multi-cast routing protocols")
    print("   • Content delivery networks")
    print("   • Bandwidth allocation across services")
    print("   • Network load balancing")
    
    print("\n4. **Resource Allocation:**")
    print("   • Cloud computing resource distribution")
    print("   • Power grid load distribution")
    print("   • Water distribution systems")
    print("   • Healthcare resource allocation")
    
    print("\n5. **Financial Systems:**")
    print("   • Multi-currency trading")
    print("   • Portfolio rebalancing")
    print("   • Risk distribution")
    print("   • Liquidity management")

def analyze_multi_flow_complexity():
    """Analyze complexity of multi-flow problems"""
    print("\n=== Multi-Flow Complexity Analysis ===")
    
    print("Problem Complexity Comparison:")
    
    print("\n1. **Single Source-Sink:**")
    print("   • Time: O(V * E²) (Edmonds-Karp)")
    print("   • Space: O(V + E)")
    print("   • Well-understood, efficient algorithms")
    
    print("\n2. **Multi-Source Multi-Sink:**")
    print("   • Time: O(max_flow_time) via transformation")
    print("   • Space: O(V + E) additional for super nodes")
    print("   • Reduces to single source-sink problem")
    
    print("\n3. **Multi-Terminal Flow:**")
    print("   • Time: O(max_flow_time)")
    print("   • Requires supply-demand balance")
    print("   • Feasibility checking important")
    
    print("\n4. **Multi-Commodity Flow:**")
    print("   • Time: O(k * max_flow_time) for k commodities")
    print("   • Space: O(k * (V + E))")
    print("   • NP-hard for integral flows")
    print("   • Polynomial for fractional flows")
    
    print("\n5. **Concurrent Flow:**")
    print("   • Time: O(log(accuracy) * max_flow_time)")
    print("   • Uses binary search on feasibility")
    print("   • Approximation algorithms available")
    
    print("\nKey Insights:")
    print("• Transformations often reduce complexity")
    print("• Multi-commodity flows are fundamentally harder")
    print("• Approximation algorithms important for large instances")
    print("• Specialized algorithms for specific network structures")

if __name__ == "__main__":
    test_multi_source_multi_sink_flow()
    demonstrate_multi_flow_applications()
    analyze_multi_flow_complexity()

"""
Multi-Source Multi-Sink Flow - Key Insights:

1. **Problem Transformations:**
   - Super source/sink reduces to standard max-flow
   - Multi-terminal requires supply-demand balance
   - Flow decomposition reveals path structure
   - Steiner tree optimization for tree-like demands

2. **Algorithm Categories:**
   - Transformation-based: Leverage existing algorithms
   - Direct methods: Specialized for multi-terminal cases
   - Approximation: For intractable variants
   - Decomposition: Understanding flow structure

3. **Complexity Considerations:**
   - Single-commodity: Polynomial time
   - Multi-commodity: NP-hard (integral), polynomial (fractional)
   - Concurrent flows: Approximation algorithms
   - Network structure affects algorithm choice

4. **Practical Applications:**
   - Supply chain and logistics optimization
   - Communication network routing
   - Resource allocation and scheduling
   - Transportation and distribution systems

5. **Implementation Strategies:**
   - Efficient graph representations
   - Careful handling of multiple commodities
   - Approximation for large-scale problems
   - Specialized algorithms for specific structures

Multi-source multi-sink flows extend basic flow theory
to handle complex real-world distribution scenarios.
"""
