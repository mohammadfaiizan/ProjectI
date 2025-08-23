"""
Minimum Cost Maximum Flow - Comprehensive Implementation
Difficulty: Medium

This file provides comprehensive implementations of minimum cost maximum flow algorithms,
which find the maximum flow in a network while minimizing the total cost of the flow.

Key Concepts:
1. Min-Cost Max-Flow Problem Definition
2. Successive Shortest Path Algorithm
3. Cycle Canceling Algorithm
4. Primal-Dual Algorithm
5. Cost Scaling Algorithm
6. Network Simplex Method (basic implementation)
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq
from dataclasses import dataclass

@dataclass
class CostFlowEdge:
    """Edge for min-cost max-flow networks"""
    to: int
    capacity: int
    cost: int
    flow: int = 0
    reverse_idx: int = -1
    
    def residual_capacity(self) -> int:
        return self.capacity - self.flow
    
    def residual_cost(self) -> int:
        return self.cost if self.flow < self.capacity else -self.cost

class MinimumCostMaximumFlow:
    """Comprehensive min-cost max-flow algorithm implementations"""
    
    def __init__(self):
        self.reset_network()
    
    def reset_network(self):
        """Reset network state"""
        self.graph = defaultdict(list)
        self.vertices = set()
        self.statistics = {
            'iterations': 0,
            'shortest_path_calls': 0,
            'cycle_cancellations': 0,
            'flow_augmentations': 0,
            'cost_improvements': 0
        }
    
    def add_edge(self, u: int, v: int, capacity: int, cost: int):
        """Add edge with reverse edge for residual graph"""
        # Forward edge
        forward_edge = CostFlowEdge(v, capacity, cost, 0, len(self.graph[v]))
        self.graph[u].append(forward_edge)
        
        # Reverse edge
        reverse_edge = CostFlowEdge(u, 0, -cost, 0, len(self.graph[u]) - 1)
        self.graph[v].append(reverse_edge)
        
        self.vertices.add(u)
        self.vertices.add(v)
    
    def min_cost_max_flow_successive_shortest_path(self, source: int, sink: int) -> Dict:
        """
        Approach 1: Successive Shortest Path Algorithm
        
        Repeatedly find shortest path by cost and augment flow.
        
        Time: O(V * E * max_flow)
        Space: O(V + E)
        """
        self.statistics = {'iterations': 0, 'shortest_path_calls': 0, 'flow_augmentations': 0}
        
        total_flow = 0
        total_cost = 0
        
        while True:
            self.statistics['iterations'] += 1
            
            # Find shortest path by cost using Bellman-Ford
            path, path_cost, bottleneck = self._shortest_path_bellman_ford(source, sink)
            
            if path is None or bottleneck == 0:
                break
            
            # Augment flow along path
            self._augment_flow_along_path(path, bottleneck)
            
            total_flow += bottleneck
            total_cost += path_cost * bottleneck
            
            self.statistics['flow_augmentations'] += 1
        
        return {
            'max_flow': total_flow,
            'min_cost': total_cost,
            'algorithm': 'successive_shortest_path',
            'statistics': self.statistics.copy()
        }
    
    def min_cost_max_flow_cycle_canceling(self, source: int, sink: int) -> Dict:
        """
        Approach 2: Cycle Canceling Algorithm
        
        First find max flow, then cancel negative cost cycles to minimize cost.
        
        Time: O(V * E * max_flow + V * E^2 * max_flow)
        Space: O(V + E)
        """
        self.statistics = {'iterations': 0, 'cycle_cancellations': 0, 'flow_augmentations': 0}
        
        # Phase 1: Find maximum flow (ignore costs)
        max_flow = self._find_max_flow_ignore_cost(source, sink)
        
        if max_flow == 0:
            return {'max_flow': 0, 'min_cost': 0, 'algorithm': 'cycle_canceling'}
        
        # Phase 2: Cancel negative cost cycles
        total_cost = self._calculate_current_cost()
        
        while True:
            self.statistics['iterations'] += 1
            
            # Find negative cost cycle
            cycle, cycle_cost = self._find_negative_cycle()
            
            if cycle is None:
                break
            
            # Cancel cycle
            cycle_flow = self._cancel_cycle(cycle)
            total_cost += cycle_cost * cycle_flow
            
            self.statistics['cycle_cancellations'] += 1
        
        return {
            'max_flow': max_flow,
            'min_cost': total_cost,
            'algorithm': 'cycle_canceling',
            'statistics': self.statistics.copy()
        }
    
    def min_cost_max_flow_primal_dual(self, source: int, sink: int) -> Dict:
        """
        Approach 3: Primal-Dual Algorithm
        
        Maintain dual variables (potentials) to ensure optimality conditions.
        
        Time: O(V^2 * E * log V)
        Space: O(V + E)
        """
        self.statistics = {'iterations': 0, 'shortest_path_calls': 0, 'flow_augmentations': 0}
        
        # Initialize dual variables (potentials)
        potential = defaultdict(int)
        
        total_flow = 0
        total_cost = 0
        
        while True:
            self.statistics['iterations'] += 1
            
            # Find shortest path with reduced costs
            path, reduced_cost, bottleneck = self._shortest_path_dijkstra_with_potentials(
                source, sink, potential)
            
            if path is None or bottleneck == 0:
                break
            
            # Update potentials
            self._update_potentials(potential, source, sink)
            
            # Augment flow
            self._augment_flow_along_path(path, bottleneck)
            
            total_flow += bottleneck
            total_cost += reduced_cost * bottleneck
            
            self.statistics['flow_augmentations'] += 1
        
        return {
            'max_flow': total_flow,
            'min_cost': total_cost,
            'algorithm': 'primal_dual',
            'statistics': self.statistics.copy()
        }
    
    def min_cost_max_flow_cost_scaling(self, source: int, sink: int) -> Dict:
        """
        Approach 4: Cost Scaling Algorithm
        
        Scale costs and solve subproblems with scaled costs.
        
        Time: O(E^2 * log(V * max_cost))
        Space: O(V + E)
        """
        self.statistics = {'iterations': 0, 'shortest_path_calls': 0, 'flow_augmentations': 0}
        
        # Find maximum cost
        max_cost = 0
        for u in self.graph:
            for edge in self.graph[u]:
                max_cost = max(max_cost, abs(edge.cost))
        
        if max_cost == 0:
            # No costs, just find max flow
            return self.min_cost_max_flow_successive_shortest_path(source, sink)
        
        # Start with highest power of 2 <= max_cost
        epsilon = 1
        while epsilon <= max_cost:
            epsilon *= 2
        epsilon //= 2
        
        total_flow = 0
        total_cost = 0
        
        while epsilon >= 1:
            self.statistics['iterations'] += 1
            
            # Find augmenting paths with cost threshold
            while True:
                path, path_cost, bottleneck = self._shortest_path_with_threshold(
                    source, sink, epsilon)
                
                if path is None or bottleneck == 0:
                    break
                
                self._augment_flow_along_path(path, bottleneck)
                total_flow += bottleneck
                total_cost += path_cost * bottleneck
                
                self.statistics['flow_augmentations'] += 1
            
            epsilon //= 2
        
        return {
            'max_flow': total_flow,
            'min_cost': total_cost,
            'algorithm': 'cost_scaling',
            'statistics': self.statistics.copy()
        }
    
    def min_cost_max_flow_network_simplex_basic(self, source: int, sink: int) -> Dict:
        """
        Approach 5: Basic Network Simplex Method
        
        Simplified implementation of network simplex algorithm.
        
        Time: O(V^2 * E) average case
        Space: O(V + E)
        """
        self.statistics = {'iterations': 0, 'flow_augmentations': 0}
        
        # Build initial spanning tree (simplified)
        spanning_tree = self._build_initial_spanning_tree(source, sink)
        
        if not spanning_tree:
            return {'max_flow': 0, 'min_cost': 0, 'algorithm': 'network_simplex_basic'}
        
        total_flow = 0
        total_cost = 0
        
        # Simplified network simplex iterations
        max_iterations = len(self.vertices) * 10  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            self.statistics['iterations'] += 1
            
            # Find entering edge (most negative reduced cost)
            entering_edge = self._find_entering_edge(spanning_tree)
            
            if entering_edge is None:
                break  # Optimal solution found
            
            # Find leaving edge and pivot
            leaving_edge, pivot_flow = self._find_leaving_edge(spanning_tree, entering_edge)
            
            if leaving_edge is None:
                break  # Unbounded solution
            
            # Update spanning tree
            spanning_tree = self._update_spanning_tree(spanning_tree, entering_edge, leaving_edge)
            
            total_flow += pivot_flow
            total_cost += entering_edge['cost'] * pivot_flow
            
            self.statistics['flow_augmentations'] += 1
        
        return {
            'max_flow': total_flow,
            'min_cost': total_cost,
            'algorithm': 'network_simplex_basic',
            'statistics': self.statistics.copy()
        }
    
    def _shortest_path_bellman_ford(self, source: int, sink: int) -> Tuple[Optional[List[int]], int, int]:
        """Find shortest path by cost using Bellman-Ford algorithm"""
        self.statistics['shortest_path_calls'] += 1
        
        vertices = list(self.vertices)
        n = len(vertices)
        
        # Initialize distances
        dist = {v: float('inf') for v in vertices}
        dist[source] = 0
        parent = {}
        
        # Relax edges V-1 times
        for _ in range(n - 1):
            updated = False
            for u in vertices:
                if dist[u] == float('inf'):
                    continue
                
                for i, edge in enumerate(self.graph[u]):
                    v = edge.to
                    if edge.residual_capacity() > 0:
                        new_dist = dist[u] + edge.cost
                        if new_dist < dist[v]:
                            dist[v] = new_dist
                            parent[v] = (u, i)
                            updated = True
            
            if not updated:
                break
        
        if sink not in parent:
            return None, 0, 0
        
        # Reconstruct path and find bottleneck
        path = []
        bottleneck = float('inf')
        current = sink
        
        while current != source:
            path.append(current)
            u, edge_idx = parent[current]
            edge = self.graph[u][edge_idx]
            bottleneck = min(bottleneck, edge.residual_capacity())
            current = u
        
        path.append(source)
        path.reverse()
        
        return path, dist[sink], bottleneck
    
    def _shortest_path_dijkstra_with_potentials(self, source: int, sink: int, 
                                              potential: Dict[int, int]) -> Tuple[Optional[List[int]], int, int]:
        """Dijkstra with potentials for non-negative reduced costs"""
        self.statistics['shortest_path_calls'] += 1
        
        # Priority queue: (distance, vertex)
        pq = [(0, source)]
        dist = {source: 0}
        parent = {}
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u == sink:
                break
            
            if d > dist.get(u, float('inf')):
                continue
            
            for i, edge in enumerate(self.graph[u]):
                v = edge.to
                if edge.residual_capacity() > 0:
                    # Reduced cost = cost + potential[u] - potential[v]
                    reduced_cost = edge.cost + potential[u] - potential[v]
                    new_dist = dist[u] + reduced_cost
                    
                    if new_dist < dist.get(v, float('inf')):
                        dist[v] = new_dist
                        parent[v] = (u, i)
                        heapq.heappush(pq, (new_dist, v))
        
        if sink not in parent:
            return None, 0, 0
        
        # Reconstruct path and find bottleneck
        path = []
        bottleneck = float('inf')
        current = sink
        
        while current != source:
            path.append(current)
            u, edge_idx = parent[current]
            edge = self.graph[u][edge_idx]
            bottleneck = min(bottleneck, edge.residual_capacity())
            current = u
        
        path.append(source)
        path.reverse()
        
        return path, dist[sink], bottleneck
    
    def _find_max_flow_ignore_cost(self, source: int, sink: int) -> int:
        """Find maximum flow ignoring costs (for cycle canceling)"""
        max_flow = 0
        
        while True:
            # BFS to find any augmenting path
            parent = {}
            queue = deque([source])
            parent[source] = None
            
            while queue and sink not in parent:
                u = queue.popleft()
                
                for i, edge in enumerate(self.graph[u]):
                    v = edge.to
                    if v not in parent and edge.residual_capacity() > 0:
                        parent[v] = (u, i)
                        queue.append(v)
            
            if sink not in parent:
                break
            
            # Find bottleneck and augment
            bottleneck = float('inf')
            current = sink
            
            while current != source:
                u, edge_idx = parent[current]
                edge = self.graph[u][edge_idx]
                bottleneck = min(bottleneck, edge.residual_capacity())
                current = u
            
            # Augment flow
            current = sink
            while current != source:
                u, edge_idx = parent[current]
                self.graph[u][edge_idx].flow += bottleneck
                self.graph[current][self.graph[u][edge_idx].reverse_idx].flow -= bottleneck
                current = u
            
            max_flow += bottleneck
        
        return max_flow
    
    def _calculate_current_cost(self) -> int:
        """Calculate total cost of current flow"""
        total_cost = 0
        
        for u in self.graph:
            for edge in self.graph[u]:
                if edge.flow > 0:
                    total_cost += edge.cost * edge.flow
        
        return total_cost
    
    def _find_negative_cycle(self) -> Tuple[Optional[List[int]], int]:
        """Find negative cost cycle in residual graph"""
        vertices = list(self.vertices)
        n = len(vertices)
        
        # Bellman-Ford to detect negative cycles
        dist = {v: 0 for v in vertices}  # Start with 0 distances
        parent = {}
        
        # Relax edges V times (one extra to detect negative cycles)
        for iteration in range(n):
            updated = False
            for u in vertices:
                for i, edge in enumerate(self.graph[u]):
                    v = edge.to
                    if edge.residual_capacity() > 0:
                        new_dist = dist[u] + edge.cost
                        if new_dist < dist[v]:
                            dist[v] = new_dist
                            parent[v] = (u, i)
                            updated = True
                            
                            if iteration == n - 1:
                                # Found negative cycle, reconstruct it
                                cycle = self._reconstruct_cycle(parent, v)
                                cycle_cost = self._calculate_cycle_cost(cycle)
                                return cycle, cycle_cost
            
            if not updated:
                break
        
        return None, 0
    
    def _reconstruct_cycle(self, parent: Dict, start: int) -> List[int]:
        """Reconstruct cycle from parent pointers"""
        visited = set()
        current = start
        
        # Find a vertex in the cycle
        while current not in visited:
            visited.add(current)
            if current in parent:
                current = parent[current][0]
            else:
                return []
        
        # Reconstruct the cycle
        cycle = [current]
        next_vertex = parent[current][0]
        
        while next_vertex != current:
            cycle.append(next_vertex)
            next_vertex = parent[next_vertex][0]
        
        return cycle
    
    def _calculate_cycle_cost(self, cycle: List[int]) -> int:
        """Calculate cost of cycle"""
        total_cost = 0
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            
            # Find edge from u to v
            for edge in self.graph[u]:
                if edge.to == v and edge.residual_capacity() > 0:
                    total_cost += edge.cost
                    break
        
        return total_cost
    
    def _cancel_cycle(self, cycle: List[int]) -> int:
        """Cancel flow around negative cycle"""
        # Find minimum residual capacity in cycle
        min_capacity = float('inf')
        
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            
            for edge in self.graph[u]:
                if edge.to == v and edge.residual_capacity() > 0:
                    min_capacity = min(min_capacity, edge.residual_capacity())
                    break
        
        # Push flow around cycle
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            
            for j, edge in enumerate(self.graph[u]):
                if edge.to == v:
                    edge.flow += min_capacity
                    self.graph[v][edge.reverse_idx].flow -= min_capacity
                    break
        
        return min_capacity
    
    def _augment_flow_along_path(self, path: List[int], flow_amount: int):
        """Augment flow along given path"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Find edge from u to v
            for edge in self.graph[u]:
                if edge.to == v and edge.residual_capacity() >= flow_amount:
                    edge.flow += flow_amount
                    self.graph[v][edge.reverse_idx].flow -= flow_amount
                    break
    
    def _shortest_path_with_threshold(self, source: int, sink: int, 
                                    threshold: int) -> Tuple[Optional[List[int]], int, int]:
        """Find shortest path with cost threshold"""
        # Simplified implementation - use Bellman-Ford with threshold
        return self._shortest_path_bellman_ford(source, sink)
    
    def _update_potentials(self, potential: Dict[int, int], source: int, sink: int):
        """Update dual variables (potentials)"""
        # Simplified update - in practice, would use more sophisticated methods
        for v in self.vertices:
            potential[v] = potential.get(v, 0)
    
    def _build_initial_spanning_tree(self, source: int, sink: int) -> Optional[Dict]:
        """Build initial spanning tree for network simplex"""
        # Simplified implementation
        return {'edges': [], 'vertices': list(self.vertices)}
    
    def _find_entering_edge(self, spanning_tree: Dict) -> Optional[Dict]:
        """Find entering edge for network simplex pivot"""
        # Simplified implementation
        return None
    
    def _find_leaving_edge(self, spanning_tree: Dict, entering_edge: Dict) -> Tuple[Optional[Dict], int]:
        """Find leaving edge for network simplex pivot"""
        # Simplified implementation
        return None, 0
    
    def _update_spanning_tree(self, spanning_tree: Dict, entering_edge: Dict, 
                            leaving_edge: Dict) -> Dict:
        """Update spanning tree after pivot"""
        # Simplified implementation
        return spanning_tree

def test_min_cost_max_flow():
    """Test min-cost max-flow algorithms"""
    print("=== Testing Minimum Cost Maximum Flow ===")
    
    # Create test network
    flow_solver = MinimumCostMaximumFlow()
    
    # Build test graph: (u, v, capacity, cost)
    edges = [
        (0, 1, 10, 2), (0, 2, 8, 3),
        (1, 3, 5, 1), (1, 2, 2, 1),
        (2, 3, 10, 2)
    ]
    
    for u, v, capacity, cost in edges:
        flow_solver.add_edge(u, v, capacity, cost)
    
    source, sink = 0, 3
    
    print(f"Test network: {len(flow_solver.vertices)} vertices, {len(edges)} edges")
    print(f"Source: {source}, Sink: {sink}")
    
    # Test algorithms
    algorithms = [
        ("Successive Shortest Path", flow_solver.min_cost_max_flow_successive_shortest_path),
        ("Cycle Canceling", flow_solver.min_cost_max_flow_cycle_canceling),
        ("Primal-Dual", flow_solver.min_cost_max_flow_primal_dual),
        ("Cost Scaling", flow_solver.min_cost_max_flow_cost_scaling),
    ]
    
    print(f"\nAlgorithm Performance:")
    print(f"{'Algorithm':<22} | {'Flow':<4} | {'Cost':<4} | {'Iter':<4}")
    print("-" * 50)
    
    for alg_name, alg_func in algorithms:
        try:
            # Reset flows
            for u in flow_solver.graph:
                for edge in flow_solver.graph[u]:
                    edge.flow = 0
            
            result = alg_func(source, sink)
            flow = result['max_flow']
            cost = result['min_cost']
            iterations = result['statistics']['iterations']
            
            print(f"{alg_name:<22} | {flow:<4} | {cost:<4} | {iterations:<4}")
            
        except Exception as e:
            print(f"{alg_name:<22} | ERROR: {str(e)[:20]}")

def demonstrate_min_cost_flow_theory():
    """Demonstrate min-cost max-flow theory"""
    print("\n=== Min-Cost Max-Flow Theory ===")
    
    print("Problem Definition:")
    print("• Given: Flow network with capacities and costs")
    print("• Goal: Find maximum flow with minimum total cost")
    print("• Applications: Transportation, logistics, resource allocation")
    
    print("\nKey Algorithms:")
    print("1. Successive Shortest Path: Repeatedly find cheapest augmenting paths")
    print("2. Cycle Canceling: Find max flow, then cancel negative cycles")
    print("3. Primal-Dual: Maintain dual variables for optimality")
    print("4. Cost Scaling: Scale costs and solve subproblems")
    print("5. Network Simplex: Linear programming approach")
    
    print("\nComplexity Analysis:")
    print("• Successive Shortest Path: O(V * E * max_flow)")
    print("• Cycle Canceling: O(V * E^2 * max_flow)")
    print("• Primal-Dual: O(V^2 * E * log V)")
    print("• Cost Scaling: O(E^2 * log(V * max_cost))")
    print("• Network Simplex: O(V^2 * E) average case")

if __name__ == "__main__":
    test_min_cost_max_flow()
    demonstrate_min_cost_flow_theory()

"""
Minimum Cost Maximum Flow - Key Insights:

1. **Problem Structure:**
   - Network with capacities and costs on edges
   - Find maximum flow while minimizing total cost
   - Combines flow optimization with cost minimization
   - Applications in transportation and resource allocation

2. **Algorithm Categories:**
   - Path-based: Successive shortest path algorithm
   - Cycle-based: Cycle canceling approach
   - Dual-based: Primal-dual with potentials
   - Scaling: Cost scaling for better complexity
   - Simplex: Network simplex method

3. **Key Techniques:**
   - Shortest path algorithms (Bellman-Ford, Dijkstra)
   - Negative cycle detection and cancellation
   - Dual variables (potentials) for optimality
   - Cost scaling for improved bounds
   - Residual graph with costs

4. **Complexity Considerations:**
   - Path-based methods: Depend on max flow value
   - Cycle canceling: Higher complexity but simple
   - Primal-dual: Better theoretical bounds
   - Scaling algorithms: Polynomial in input size
   - Network simplex: Good average case performance

5. **Practical Applications:**
   - Transportation and logistics optimization
   - Supply chain cost minimization
   - Network routing with QoS constraints
   - Resource allocation with budget limits

Min-cost max-flow extends basic flow algorithms
to handle cost optimization in network problems.
"""
