"""
Advanced Network Flow - Comprehensive Implementation
Difficulty: Hard

This file provides advanced network flow algorithms and techniques including
Dinic's algorithm, push-relabel with FIFO/highest-label selection, capacity scaling,
and specialized flow algorithms for complex scenarios.

Key Concepts:
1. Dinic's Algorithm with Blocking Flows
2. Push-Relabel with Advanced Selection Rules
3. Capacity Scaling and Shortest Augmenting Path
4. Multi-commodity Flow
5. Flow with Lower Bounds
6. Parametric Maximum Flow
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
import math
from dataclasses import dataclass

@dataclass
class FlowEdge:
    """Enhanced flow edge with additional properties"""
    to: int
    capacity: int
    flow: int = 0
    cost: int = 0
    reverse_idx: int = -1
    
    def residual_capacity(self) -> int:
        return self.capacity - self.flow

class AdvancedNetworkFlow:
    """Advanced network flow algorithm implementations"""
    
    def __init__(self):
        self.reset_network()
    
    def reset_network(self):
        """Reset network state"""
        self.graph = defaultdict(list)
        self.vertices = set()
        self.statistics = {
            'phases': 0,
            'blocking_flows': 0,
            'pushes': 0,
            'relabels': 0,
            'gap_optimizations': 0
        }
    
    def add_edge(self, u: int, v: int, capacity: int, cost: int = 0):
        """Add edge with reverse edge for residual graph"""
        # Forward edge
        forward_edge = FlowEdge(v, capacity, 0, cost, len(self.graph[v]))
        self.graph[u].append(forward_edge)
        
        # Reverse edge
        reverse_edge = FlowEdge(u, 0, 0, -cost, len(self.graph[u]) - 1)
        self.graph[v].append(reverse_edge)
        
        self.vertices.add(u)
        self.vertices.add(v)
    
    def max_flow_dinic(self, source: int, sink: int) -> Dict:
        """
        Approach 1: Dinic's Algorithm
        
        Advanced max-flow algorithm using level graphs and blocking flows.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        self.statistics = {'phases': 0, 'blocking_flows': 0, 'pushes': 0, 'relabels': 0}
        
        def bfs_build_level_graph():
            """Build level graph using BFS"""
            level = {source: 0}
            queue = deque([source])
            
            while queue:
                u = queue.popleft()
                
                for edge in self.graph[u]:
                    v = edge.to
                    if v not in level and edge.residual_capacity() > 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            
            return level if sink in level else None
        
        def dfs_blocking_flow(u: int, sink: int, level: Dict[int, int], 
                             pushed: int, blocked: Set[int]) -> int:
            """Find blocking flow using DFS"""
            if u == sink:
                return pushed
            
            if u in blocked:
                return 0
            
            total_flow = 0
            
            for i, edge in enumerate(self.graph[u]):
                v = edge.to
                
                if (v in level and level[v] == level[u] + 1 and 
                    edge.residual_capacity() > 0 and v not in blocked):
                    
                    bottleneck = min(pushed - total_flow, edge.residual_capacity())
                    flow = dfs_blocking_flow(v, sink, level, bottleneck, blocked)
                    
                    if flow > 0:
                        edge.flow += flow
                        self.graph[v][edge.reverse_idx].flow -= flow
                        total_flow += flow
                        self.statistics['pushes'] += 1
                        
                        if total_flow == pushed:
                            break
            
            if total_flow == 0:
                blocked.add(u)
            
            return total_flow
        
        max_flow = 0
        
        while True:
            self.statistics['phases'] += 1
            
            # Build level graph
            level = bfs_build_level_graph()
            if level is None:
                break
            
            # Find blocking flow
            blocked = set()
            while True:
                flow = dfs_blocking_flow(source, sink, level, float('inf'), blocked)
                if flow == 0:
                    break
                max_flow += flow
                self.statistics['blocking_flows'] += 1
        
        return {
            'max_flow': max_flow,
            'algorithm': 'dinic',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_push_relabel_fifo(self, source: int, sink: int) -> Dict:
        """
        Approach 2: Push-Relabel with FIFO Selection
        
        Push-relabel algorithm with FIFO active vertex selection.
        
        Time: O(V^3)
        Space: O(V + E)
        """
        self.statistics = {'pushes': 0, 'relabels': 0, 'gap_optimizations': 0}
        
        vertices = list(self.vertices)
        n = len(vertices)
        
        # Initialize
        excess = defaultdict(int)
        height = defaultdict(int)
        height[source] = n
        
        # Create initial preflow
        for edge in self.graph[source]:
            if edge.residual_capacity() > 0:
                flow_amount = edge.residual_capacity()
                edge.flow = flow_amount
                self.graph[edge.to][edge.reverse_idx].flow = -flow_amount
                excess[edge.to] += flow_amount
                excess[source] -= flow_amount
        
        # FIFO queue for active vertices
        active_queue = deque()
        in_queue = set()
        
        for v in vertices:
            if v != source and v != sink and excess[v] > 0:
                active_queue.append(v)
                in_queue.add(v)
        
        def push(u: int, edge: FlowEdge) -> bool:
            """Push flow along edge"""
            if edge.residual_capacity() <= 0 or height[u] != height[edge.to] + 1:
                return False
            
            flow_amount = min(excess[u], edge.residual_capacity())
            edge.flow += flow_amount
            self.graph[edge.to][edge.reverse_idx].flow -= flow_amount
            
            excess[u] -= flow_amount
            excess[edge.to] += flow_amount
            
            # Add to queue if became active
            if (edge.to != source and edge.to != sink and 
                edge.to not in in_queue and excess[edge.to] > 0):
                active_queue.append(edge.to)
                in_queue.add(edge.to)
            
            self.statistics['pushes'] += 1
            return True
        
        def relabel(u: int):
            """Relabel vertex u"""
            min_height = float('inf')
            
            for edge in self.graph[u]:
                if edge.residual_capacity() > 0:
                    min_height = min(min_height, height[edge.to])
            
            if min_height < float('inf'):
                height[u] = min_height + 1
                self.statistics['relabels'] += 1
        
        def gap_optimization():
            """Gap optimization heuristic"""
            height_count = defaultdict(int)
            for v in vertices:
                height_count[height[v]] += 1
            
            for h in range(n):
                if height_count[h] == 0:
                    # Gap found, relabel all vertices above gap
                    for v in vertices:
                        if height[v] > h:
                            height[v] = max(height[v], n + 1)
                    self.statistics['gap_optimizations'] += 1
                    break
        
        # Main algorithm loop
        while active_queue:
            u = active_queue.popleft()
            in_queue.remove(u)
            
            old_height = height[u]
            
            # Try to push from u
            pushed = False
            for edge in self.graph[u]:
                if excess[u] <= 0:
                    break
                if push(u, edge):
                    pushed = True
            
            # If couldn't push all excess, relabel
            if excess[u] > 0:
                relabel(u)
                if height[u] > old_height:
                    active_queue.appendleft(u)  # High priority
                    in_queue.add(u)
            
            # Periodic gap optimization
            if self.statistics['relabels'] % n == 0:
                gap_optimization()
        
        max_flow = sum(edge.flow for edge in self.graph[source])
        
        return {
            'max_flow': max_flow,
            'algorithm': 'push_relabel_fifo',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_push_relabel_highest_label(self, source: int, sink: int) -> Dict:
        """
        Approach 3: Push-Relabel with Highest Label Selection
        
        Push-relabel with highest-label first selection rule.
        
        Time: O(V^2 * sqrt(E))
        Space: O(V + E)
        """
        self.statistics = {'pushes': 0, 'relabels': 0}
        
        vertices = list(self.vertices)
        n = len(vertices)
        
        # Initialize
        excess = defaultdict(int)
        height = defaultdict(int)
        height[source] = n
        
        # Create initial preflow
        for edge in self.graph[source]:
            if edge.residual_capacity() > 0:
                flow_amount = edge.residual_capacity()
                edge.flow = flow_amount
                self.graph[edge.to][edge.reverse_idx].flow = -flow_amount
                excess[edge.to] += flow_amount
                excess[source] -= flow_amount
        
        # Priority queue for active vertices (max-heap by height)
        active_heap = []
        
        for v in vertices:
            if v != source and v != sink and excess[v] > 0:
                heapq.heappush(active_heap, (-height[v], v))
        
        def push(u: int, edge: FlowEdge) -> bool:
            """Push flow along edge"""
            if edge.residual_capacity() <= 0 or height[u] != height[edge.to] + 1:
                return False
            
            flow_amount = min(excess[u], edge.residual_capacity())
            edge.flow += flow_amount
            self.graph[edge.to][edge.reverse_idx].flow -= flow_amount
            
            excess[u] -= flow_amount
            excess[edge.to] += flow_amount
            
            # Add to heap if became active
            if (edge.to != source and edge.to != sink and excess[edge.to] == flow_amount):
                heapq.heappush(active_heap, (-height[edge.to], edge.to))
            
            self.statistics['pushes'] += 1
            return True
        
        def relabel(u: int):
            """Relabel vertex u"""
            min_height = float('inf')
            
            for edge in self.graph[u]:
                if edge.residual_capacity() > 0:
                    min_height = min(min_height, height[edge.to])
            
            if min_height < float('inf'):
                height[u] = min_height + 1
                self.statistics['relabels'] += 1
        
        # Main algorithm loop
        while active_heap:
            neg_height, u = heapq.heappop(active_heap)
            
            if excess[u] <= 0 or -neg_height != height[u]:
                continue  # Outdated entry
            
            # Try to push from u
            for edge in self.graph[u]:
                if excess[u] <= 0:
                    break
                push(u, edge)
            
            # If still has excess, relabel and re-add
            if excess[u] > 0:
                relabel(u)
                heapq.heappush(active_heap, (-height[u], u))
        
        max_flow = sum(edge.flow for edge in self.graph[source])
        
        return {
            'max_flow': max_flow,
            'algorithm': 'push_relabel_highest_label',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_capacity_scaling_advanced(self, source: int, sink: int) -> Dict:
        """
        Approach 4: Advanced Capacity Scaling
        
        Capacity scaling with improved scaling strategy and optimizations.
        
        Time: O(E^2 * log(max_capacity))
        Space: O(V + E)
        """
        self.statistics = {'phases': 0, 'pushes': 0}
        
        # Find maximum capacity
        max_capacity = 0
        for u in self.graph:
            for edge in self.graph[u]:
                max_capacity = max(max_capacity, edge.capacity)
        
        if max_capacity == 0:
            return {'max_flow': 0, 'algorithm': 'capacity_scaling_advanced'}
        
        # Start with highest power of 2 <= max_capacity
        delta = 1
        while delta <= max_capacity:
            delta *= 2
        delta //= 2
        
        max_flow = 0
        
        while delta >= 1:
            self.statistics['phases'] += 1
            
            # Find augmenting paths with capacity >= delta
            while True:
                # BFS to find path with sufficient capacity
                parent = {}
                queue = deque([source])
                parent[source] = None
                
                found_path = False
                
                while queue and not found_path:
                    u = queue.popleft()
                    
                    for edge in self.graph[u]:
                        v = edge.to
                        if v not in parent and edge.residual_capacity() >= delta:
                            parent[v] = (u, edge)
                            queue.append(v)
                            
                            if v == sink:
                                found_path = True
                                break
                
                if not found_path:
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
                    self.graph[current][edge.reverse_idx].flow -= bottleneck
                    current = u
                
                max_flow += bottleneck
                self.statistics['pushes'] += 1
            
            delta //= 2
        
        return {
            'max_flow': max_flow,
            'algorithm': 'capacity_scaling_advanced',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_with_lower_bounds(self, source: int, sink: int, 
                                  lower_bounds: Dict[Tuple[int, int], int]) -> Dict:
        """
        Approach 5: Maximum Flow with Lower Bounds
        
        Solve max-flow problem with lower bound constraints on edges.
        
        Time: O(max_flow_time)
        Space: O(V + E)
        """
        # Transform to standard max-flow problem
        # Add super source and super sink
        super_source = max(self.vertices) + 1
        super_sink = super_source + 1
        
        # Adjust capacities and add circulation edges
        circulation_demand = defaultdict(int)
        
        # Process lower bounds
        for (u, v), lower_bound in lower_bounds.items():
            # Find corresponding edge
            for edge in self.graph[u]:
                if edge.to == v:
                    edge.capacity -= lower_bound
                    circulation_demand[u] -= lower_bound
                    circulation_demand[v] += lower_bound
                    break
        
        # Add super source/sink edges for circulation
        for vertex, demand in circulation_demand.items():
            if demand > 0:
                self.add_edge(super_source, vertex, demand)
            elif demand < 0:
                self.add_edge(vertex, super_sink, -demand)
        
        # Add edge from sink to source with infinite capacity
        self.add_edge(sink, source, float('inf'))
        
        # Solve circulation problem
        circulation_result = self.max_flow_dinic(super_source, super_sink)
        
        # Check if circulation is feasible
        required_circulation = sum(max(0, d) for d in circulation_demand.values())
        
        if circulation_result['max_flow'] < required_circulation:
            return {'feasible': False, 'algorithm': 'flow_with_lower_bounds'}
        
        # Find maximum flow in original problem
        # Remove super source/sink edges and sink-to-source edge
        self.graph[super_source].clear()
        self.graph[super_sink].clear()
        
        for edge in self.graph[sink]:
            if edge.to == source:
                edge.capacity = 0
                break
        
        # Solve original max-flow
        original_result = self.max_flow_dinic(source, sink)
        
        return {
            'feasible': True,
            'max_flow': original_result['max_flow'],
            'algorithm': 'flow_with_lower_bounds',
            'circulation_flow': circulation_result['max_flow']
        }
    
    def parametric_max_flow(self, source: int, sink: int, 
                           parameter_edges: List[Tuple[int, int, callable]]) -> Dict:
        """
        Approach 6: Parametric Maximum Flow
        
        Solve max-flow as function of parameter λ.
        
        Time: O(k * max_flow_time) where k is number of breakpoints
        Space: O(V + E)
        """
        # Find breakpoints where flow value changes
        breakpoints = set([0])
        
        # Analyze parameter functions to find critical points
        for u, v, capacity_func in parameter_edges:
            # Sample function to find approximate breakpoints
            for lambda_val in range(0, 101):
                try:
                    capacity = capacity_func(lambda_val)
                    if capacity >= 0:
                        breakpoints.add(lambda_val)
                except:
                    continue
        
        breakpoints = sorted(breakpoints)
        flow_function = []
        
        for lambda_val in breakpoints:
            # Set capacities based on parameter
            for u, v, capacity_func in parameter_edges:
                capacity = capacity_func(lambda_val)
                
                # Find and update corresponding edge
                for edge in self.graph[u]:
                    if edge.to == v:
                        edge.capacity = max(0, capacity)
                        break
            
            # Compute max-flow for this parameter value
            result = self.max_flow_dinic(source, sink)
            flow_function.append((lambda_val, result['max_flow']))
            
            # Reset flows for next iteration
            for u in self.graph:
                for edge in self.graph[u]:
                    edge.flow = 0
        
        return {
            'flow_function': flow_function,
            'breakpoints': breakpoints,
            'algorithm': 'parametric_max_flow'
        }

def test_advanced_network_flow():
    """Test advanced network flow algorithms"""
    print("=== Testing Advanced Network Flow Algorithms ===")
    
    # Create test network
    flow_solver = AdvancedNetworkFlow()
    
    # Build test graph
    edges = [
        (0, 1, 16), (0, 2, 13),
        (1, 2, 10), (1, 3, 12),
        (2, 1, 4), (2, 4, 14),
        (3, 2, 9), (3, 5, 20),
        (4, 3, 7), (4, 5, 4)
    ]
    
    for u, v, capacity in edges:
        flow_solver.add_edge(u, v, capacity)
    
    source, sink = 0, 5
    expected_flow = 23
    
    print(f"Test network: {len(flow_solver.vertices)} vertices, {len(edges)} edges")
    print(f"Source: {source}, Sink: {sink}, Expected flow: {expected_flow}")
    
    # Test algorithms
    algorithms = [
        ("Dinic's Algorithm", flow_solver.max_flow_dinic),
        ("Push-Relabel FIFO", flow_solver.max_flow_push_relabel_fifo),
        ("Push-Relabel Highest", flow_solver.max_flow_push_relabel_highest_label),
        ("Capacity Scaling", flow_solver.max_flow_capacity_scaling_advanced),
    ]
    
    print(f"\nAlgorithm Performance:")
    print(f"{'Algorithm':<20} | {'Flow':<4} | {'Phases':<6} | {'Pushes':<6}")
    print("-" * 50)
    
    for alg_name, alg_func in algorithms:
        try:
            # Reset network state
            for u in flow_solver.graph:
                for edge in flow_solver.graph[u]:
                    edge.flow = 0
            
            result = alg_func(source, sink)
            flow = result['max_flow']
            stats = result['statistics']
            
            phases = stats.get('phases', stats.get('relabels', 0))
            pushes = stats.get('pushes', 0)
            
            status = "✓" if flow == expected_flow else "✗"
            print(f"{alg_name:<20} | {flow:<4} | {phases:<6} | {pushes:<6} {status}")
            
        except Exception as e:
            print(f"{alg_name:<20} | ERROR: {str(e)[:20]}")

def demonstrate_advanced_techniques():
    """Demonstrate advanced flow techniques"""
    print("\n=== Advanced Flow Techniques Demo ===")
    
    print("1. **Dinic's Algorithm Advantages:**")
    print("   • Uses level graphs to guide search")
    print("   • Blocking flows ensure progress")
    print("   • O(V²E) time complexity")
    print("   • Excellent for unit capacity networks")
    
    print("\n2. **Push-Relabel Optimizations:**")
    print("   • FIFO selection: Simple and effective")
    print("   • Highest-label: Better theoretical bounds")
    print("   • Gap optimization: Prune unreachable vertices")
    print("   • Global relabeling: Periodic BFS updates")
    
    print("\n3. **Capacity Scaling Benefits:**")
    print("   • Focuses on high-capacity paths first")
    print("   • Polynomial time guarantee")
    print("   • Good for networks with large capacity ranges")
    print("   • Can be combined with other techniques")
    
    print("\n4. **Specialized Flow Problems:**")
    print("   • Lower bounds: Circulation-based transformation")
    print("   • Multi-commodity: Multiple flow types")
    print("   • Parametric: Flow as function of parameter")
    print("   • Dynamic: Time-varying capacities")
    
    print("\n5. **Implementation Considerations:**")
    print("   • Data structure choice affects performance")
    print("   • Numerical precision for real-valued capacities")
    print("   • Memory optimization for large networks")
    print("   • Parallelization opportunities")

def analyze_algorithm_selection():
    """Analyze when to use different advanced algorithms"""
    print("\n=== Algorithm Selection Guide ===")
    
    print("Network Characteristics vs Algorithm Choice:")
    
    print("\n1. **Graph Density:**")
    print("   • Sparse (E ≈ V): Dinic's, Capacity Scaling")
    print("   • Dense (E ≈ V²): Push-Relabel variants")
    print("   • Very dense: Matrix-based implementations")
    
    print("\n2. **Capacity Distribution:**")
    print("   • Uniform capacities: Standard algorithms")
    print("   • Wide range: Capacity scaling")
    print("   • Unit capacities: Specialized unit-capacity algorithms")
    print("   • Integer vs real: Affects precision requirements")
    
    print("\n3. **Network Structure:**")
    print("   • Layered graphs: Dinic's algorithm")
    print("   • Planar graphs: Specialized planar algorithms")
    print("   • Bipartite graphs: Hungarian algorithm variants")
    print("   • Trees: Linear-time algorithms")
    
    print("\n4. **Problem Constraints:**")
    print("   • Lower bounds: Circulation-based approach")
    print("   • Multiple sources/sinks: Virtual super nodes")
    print("   • Node capacities: Node splitting transformation")
    print("   • Costs: Min-cost max-flow algorithms")
    
    print("\n5. **Performance Requirements:**")
    print("   • Real-time: Approximation algorithms")
    print("   • Batch processing: Exact algorithms")
    print("   • Memory constrained: Streaming algorithms")
    print("   • Parallel processing: Parallelizable variants")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of advanced flow algorithms"""
    print("\n=== Real-World Applications ===")
    
    print("Advanced Flow Algorithm Applications:")
    
    print("\n1. **Network Design and Optimization:**")
    print("   • Internet backbone capacity planning")
    print("   • Transportation network optimization")
    print("   • Power grid load balancing")
    print("   • Water distribution systems")
    
    print("\n2. **Resource Allocation:**")
    print("   • Cloud computing resource scheduling")
    print("   • Manufacturing process optimization")
    print("   • Supply chain management")
    print("   • Project resource allocation")
    
    print("\n3. **Computer Vision and Graphics:**")
    print("   • Image segmentation using graph cuts")
    print("   • Stereo vision correspondence")
    print("   • Medical image analysis")
    print("   • Video processing and compression")
    
    print("\n4. **Bioinformatics:**")
    print("   • Protein folding prediction")
    print("   • Gene regulatory network analysis")
    print("   • Phylogenetic tree construction")
    print("   • Drug discovery optimization")
    
    print("\n5. **Financial and Economic Modeling:**")
    print("   • Portfolio optimization")
    print("   • Risk management")
    print("   • Market clearing mechanisms")
    print("   • Auction design")
    
    print("\nKey Success Factors:")
    print("• Problem modeling as flow network")
    print("• Algorithm selection based on network properties")
    print("• Efficient implementation and optimization")
    print("• Handling of real-world constraints and objectives")

if __name__ == "__main__":
    test_advanced_network_flow()
    demonstrate_advanced_techniques()
    analyze_algorithm_selection()
    demonstrate_real_world_applications()

"""
Advanced Network Flow - Key Insights:

1. **Algorithm Evolution:**
   - Ford-Fulkerson → Edmonds-Karp → Dinic's → Push-Relabel
   - Each generation improves worst-case complexity
   - Specialized algorithms for specific network types
   - Modern implementations combine multiple techniques

2. **Performance Optimization:**
   - Level graphs guide search efficiently
   - Priority-based vertex selection
   - Gap optimization and global relabeling
   - Capacity scaling for wide capacity ranges

3. **Problem Extensions:**
   - Lower bound constraints via circulation
   - Multi-commodity flows for multiple flow types
   - Parametric flows for sensitivity analysis
   - Dynamic flows for time-varying networks

4. **Implementation Strategies:**
   - Efficient data structures (adjacency lists, heaps)
   - Numerical stability considerations
   - Memory optimization techniques
   - Parallelization opportunities

5. **Application Domains:**
   - Network infrastructure optimization
   - Resource allocation and scheduling
   - Computer vision and image processing
   - Bioinformatics and computational biology

Advanced flow algorithms provide the foundation for
solving complex optimization problems in networked systems.
"""
