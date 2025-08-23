"""
Maximum Flow with Lower Bounds - Comprehensive Implementation
Difficulty: Hard

This file provides comprehensive implementations for maximum flow problems with
lower bound constraints on edges, which extends the classical max-flow problem
by requiring minimum flow amounts on certain edges.

Key Concepts:
1. Flow Networks with Lower Bounds
2. Circulation Problem Reduction
3. Feasibility Testing
4. Super Source and Super Sink Construction
5. Flow Decomposition with Constraints
6. Applications in Resource Allocation
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq

class MaximumFlowWithLowerBounds:
    """Comprehensive implementation of max flow with lower bounds"""
    
    def __init__(self):
        self.reset_network()
    
    def reset_network(self):
        """Reset network state"""
        self.graph = defaultdict(lambda: defaultdict(int))
        self.lower_bounds = defaultdict(lambda: defaultdict(int))
        self.vertices = set()
        self.statistics = {
            'circulation_flow': 0,
            'max_flow_iterations': 0,
            'feasibility_checks': 0,
            'flow_augmentations': 0
        }
    
    def add_edge(self, u: int, v: int, capacity: int, lower_bound: int = 0):
        """Add edge with capacity and lower bound constraints"""
        if lower_bound > capacity:
            raise ValueError("Lower bound cannot exceed capacity")
        
        self.graph[u][v] = capacity
        self.lower_bounds[u][v] = lower_bound
        self.vertices.add(u)
        self.vertices.add(v)
    
    def max_flow_with_lower_bounds_circulation(self, source: int, sink: int) -> Dict:
        """
        Approach 1: Circulation-Based Method
        
        Transform to circulation problem by adding super source and super sink.
        
        Time: O(V * E^2) - same as max flow
        Space: O(V + E)
        """
        self.statistics = {'circulation_flow': 0, 'max_flow_iterations': 0, 'feasibility_checks': 0}
        
        # Step 1: Check if lower bounds are satisfiable
        if not self._check_lower_bound_feasibility():
            return {
                'feasible': False,
                'max_flow': 0,
                'reason': 'Lower bounds not satisfiable',
                'algorithm': 'circulation_based'
            }
        
        # Step 2: Create circulation network
        circulation_graph, super_source, super_sink = self._create_circulation_network(source, sink)
        
        # Step 3: Solve circulation problem
        circulation_flow = self._solve_circulation(circulation_graph, super_source, super_sink)
        
        # Step 4: Check if circulation is feasible
        required_circulation = self._calculate_required_circulation()
        
        if circulation_flow < required_circulation:
            return {
                'feasible': False,
                'max_flow': 0,
                'circulation_flow': circulation_flow,
                'required_circulation': required_circulation,
                'algorithm': 'circulation_based'
            }
        
        # Step 5: Find maximum flow in transformed network
        max_flow = self._find_max_flow_after_circulation(source, sink)
        
        return {
            'feasible': True,
            'max_flow': max_flow,
            'circulation_flow': circulation_flow,
            'algorithm': 'circulation_based',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_with_lower_bounds_direct(self, source: int, sink: int) -> Dict:
        """
        Approach 2: Direct Transformation Method
        
        Directly transform the network by adjusting capacities and demands.
        
        Time: O(V * E^2)
        Space: O(V + E)
        """
        self.statistics = {'max_flow_iterations': 0, 'flow_augmentations': 0}
        
        # Create transformed network
        transformed_graph = defaultdict(lambda: defaultdict(int))
        demand = defaultdict(int)
        
        # Transform edges: new_capacity = original_capacity - lower_bound
        for u in self.graph:
            for v in self.graph[u]:
                capacity = self.graph[u][v]
                lower_bound = self.lower_bounds[u][v]
                
                # Add transformed edge
                transformed_graph[u][v] = capacity - lower_bound
                
                # Update demands
                demand[u] -= lower_bound
                demand[v] += lower_bound
        
        # Create super source and super sink for demand satisfaction
        super_source = max(self.vertices) + 1
        super_sink = super_source + 1
        
        total_demand = 0
        
        # Connect super source to vertices with positive demand
        for vertex in self.vertices:
            if demand[vertex] > 0:
                transformed_graph[super_source][vertex] = demand[vertex]
                total_demand += demand[vertex]
            elif demand[vertex] < 0:
                transformed_graph[vertex][super_sink] = -demand[vertex]
        
        # Add edge from sink to source with infinite capacity
        transformed_graph[sink][source] = float('inf')
        
        # Solve circulation problem
        circulation_flow = self._edmonds_karp_max_flow(transformed_graph, super_source, super_sink)
        
        if circulation_flow < total_demand:
            return {
                'feasible': False,
                'max_flow': 0,
                'algorithm': 'direct_transformation'
            }
        
        # Remove sink-to-source edge and find max flow
        transformed_graph[sink][source] = 0
        max_flow = self._edmonds_karp_max_flow(transformed_graph, source, sink)
        
        return {
            'feasible': True,
            'max_flow': max_flow,
            'algorithm': 'direct_transformation',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_with_lower_bounds_preflow_push(self, source: int, sink: int) -> Dict:
        """
        Approach 3: Preflow-Push with Lower Bounds
        
        Adapt preflow-push algorithm to handle lower bound constraints.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        self.statistics = {'flow_augmentations': 0, 'feasibility_checks': 0}
        
        # Initialize preflow with lower bounds
        flow = defaultdict(lambda: defaultdict(int))
        excess = defaultdict(int)
        height = defaultdict(int)
        
        # Set initial flow to lower bounds
        for u in self.graph:
            for v in self.graph[u]:
                lower_bound = self.lower_bounds[u][v]
                flow[u][v] = lower_bound
                excess[u] -= lower_bound
                excess[v] += lower_bound
        
        # Check initial feasibility
        if not self._check_preflow_feasibility(excess, source, sink):
            return {
                'feasible': False,
                'max_flow': 0,
                'algorithm': 'preflow_push'
            }
        
        # Initialize height function
        height[source] = len(self.vertices)
        
        # Preflow-push algorithm
        max_flow = self._preflow_push_with_bounds(flow, excess, height, source, sink)
        
        return {
            'feasible': True,
            'max_flow': max_flow,
            'algorithm': 'preflow_push',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_with_lower_bounds_scaling(self, source: int, sink: int) -> Dict:
        """
        Approach 4: Capacity Scaling with Lower Bounds
        
        Use capacity scaling approach adapted for lower bound constraints.
        
        Time: O(E^2 * log(max_capacity))
        Space: O(V + E)
        """
        self.statistics = {'max_flow_iterations': 0, 'flow_augmentations': 0}
        
        # Find maximum capacity
        max_capacity = 0
        for u in self.graph:
            for v in self.graph[u]:
                max_capacity = max(max_capacity, self.graph[u][v])
        
        # Start with highest power of 2 <= max_capacity
        delta = 1
        while delta <= max_capacity:
            delta *= 2
        delta //= 2
        
        # Initialize flow with lower bounds
        current_flow = defaultdict(lambda: defaultdict(int))
        for u in self.graph:
            for v in self.graph[u]:
                current_flow[u][v] = self.lower_bounds[u][v]
        
        total_flow = sum(current_flow[source][v] for v in self.graph[source])
        
        while delta >= 1:
            self.statistics['max_flow_iterations'] += 1
            
            # Find augmenting paths with capacity >= delta
            while True:
                path, bottleneck = self._find_augmenting_path_with_threshold(
                    source, sink, delta, current_flow)
                
                if path is None or bottleneck == 0:
                    break
                
                # Augment flow
                self._augment_flow_along_path(path, bottleneck, current_flow)
                total_flow += bottleneck
                
                self.statistics['flow_augmentations'] += 1
            
            delta //= 2
        
        return {
            'feasible': True,
            'max_flow': total_flow,
            'algorithm': 'capacity_scaling',
            'statistics': self.statistics.copy()
        }
    
    def max_flow_with_lower_bounds_decomposition(self, source: int, sink: int) -> Dict:
        """
        Approach 5: Flow Decomposition Method
        
        Decompose the problem into feasible flow + additional flow.
        
        Time: O(V * E^2)
        Space: O(V + E)
        """
        self.statistics = {'feasibility_checks': 0, 'flow_augmentations': 0}
        
        # Step 1: Find a feasible flow satisfying lower bounds
        feasible_flow = self._find_feasible_flow(source, sink)
        
        if feasible_flow is None:
            return {
                'feasible': False,
                'max_flow': 0,
                'algorithm': 'flow_decomposition'
            }
        
        # Step 2: Create residual network with feasible flow
        residual_graph = self._create_residual_network_with_flow(feasible_flow)
        
        # Step 3: Find additional flow in residual network
        additional_flow = self._edmonds_karp_max_flow(residual_graph, source, sink)
        
        # Step 4: Calculate total flow
        base_flow = sum(feasible_flow[source][v] for v in feasible_flow[source])
        total_flow = base_flow + additional_flow
        
        return {
            'feasible': True,
            'max_flow': total_flow,
            'base_flow': base_flow,
            'additional_flow': additional_flow,
            'algorithm': 'flow_decomposition',
            'statistics': self.statistics.copy()
        }
    
    def _check_lower_bound_feasibility(self) -> bool:
        """Check if lower bounds are satisfiable using flow conservation"""
        self.statistics['feasibility_checks'] += 1
        
        # Calculate net demand for each vertex
        net_demand = defaultdict(int)
        
        for u in self.graph:
            for v in self.graph[u]:
                lower_bound = self.lower_bounds[u][v]
                net_demand[u] -= lower_bound
                net_demand[v] += lower_bound
        
        # Check if total demand is zero (necessary condition)
        total_demand = sum(net_demand.values())
        return abs(total_demand) < 1e-9
    
    def _create_circulation_network(self, source: int, sink: int) -> Tuple[Dict, int, int]:
        """Create circulation network for lower bound problem"""
        circulation_graph = defaultdict(lambda: defaultdict(int))
        
        # Copy original edges with adjusted capacities
        for u in self.graph:
            for v in self.graph[u]:
                capacity = self.graph[u][v]
                lower_bound = self.lower_bounds[u][v]
                circulation_graph[u][v] = capacity - lower_bound
        
        # Create super source and super sink
        super_source = max(self.vertices) + 1
        super_sink = super_source + 1
        
        # Calculate demands
        demand = defaultdict(int)
        for u in self.graph:
            for v in self.graph[u]:
                lower_bound = self.lower_bounds[u][v]
                demand[u] -= lower_bound
                demand[v] += lower_bound
        
        # Connect super source and super sink
        for vertex in self.vertices:
            if demand[vertex] > 0:
                circulation_graph[super_source][vertex] = demand[vertex]
            elif demand[vertex] < 0:
                circulation_graph[vertex][super_sink] = -demand[vertex]
        
        # Add edge from sink to source
        circulation_graph[sink][source] = float('inf')
        
        return circulation_graph, super_source, super_sink
    
    def _solve_circulation(self, graph: Dict, super_source: int, super_sink: int) -> int:
        """Solve circulation problem using max flow"""
        return self._edmonds_karp_max_flow(graph, super_source, super_sink)
    
    def _calculate_required_circulation(self) -> int:
        """Calculate required circulation flow"""
        total_demand = 0
        
        for u in self.graph:
            for v in self.graph[u]:
                lower_bound = self.lower_bounds[u][v]
                total_demand += lower_bound
        
        return total_demand
    
    def _find_max_flow_after_circulation(self, source: int, sink: int) -> int:
        """Find max flow after establishing circulation"""
        # Create residual graph after circulation
        residual_graph = defaultdict(lambda: defaultdict(int))
        
        for u in self.graph:
            for v in self.graph[u]:
                capacity = self.graph[u][v]
                lower_bound = self.lower_bounds[u][v]
                residual_graph[u][v] = capacity - lower_bound
        
        return self._edmonds_karp_max_flow(residual_graph, source, sink)
    
    def _edmonds_karp_max_flow(self, graph: Dict, source: int, sink: int) -> int:
        """Standard Edmonds-Karp max flow algorithm"""
        max_flow = 0
        
        # Create residual graph
        residual = defaultdict(lambda: defaultdict(int))
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
        
        while True:
            self.statistics['max_flow_iterations'] += 1
            
            # BFS to find augmenting path
            parent = {}
            queue = deque([source])
            parent[source] = None
            
            while queue and sink not in parent:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if neighbor not in parent and residual[current][neighbor] > 0:
                        parent[neighbor] = current
                        queue.append(neighbor)
            
            if sink not in parent:
                break
            
            # Find bottleneck capacity
            path_flow = float('inf')
            current = sink
            
            while current != source:
                prev = parent[current]
                path_flow = min(path_flow, residual[prev][current])
                current = prev
            
            # Update residual graph
            current = sink
            while current != source:
                prev = parent[current]
                residual[prev][current] -= path_flow
                residual[current][prev] += path_flow
                current = prev
            
            max_flow += path_flow
        
        return max_flow
    
    def _check_preflow_feasibility(self, excess: Dict, source: int, sink: int) -> bool:
        """Check if initial preflow is feasible"""
        # All vertices except source and sink should have non-negative excess
        for vertex in self.vertices:
            if vertex != source and vertex != sink and excess[vertex] < 0:
                return False
        
        return True
    
    def _preflow_push_with_bounds(self, flow: Dict, excess: Dict, height: Dict, 
                                 source: int, sink: int) -> int:
        """Preflow-push algorithm adapted for lower bounds"""
        vertices = list(self.vertices)
        
        def push(u: int, v: int) -> bool:
            """Push flow from u to v"""
            if (excess[u] > 0 and height[u] == height[v] + 1 and 
                flow[u][v] < self.graph[u][v]):
                
                push_amount = min(excess[u], self.graph[u][v] - flow[u][v])
                flow[u][v] += push_amount
                flow[v][u] -= push_amount
                excess[u] -= push_amount
                excess[v] += push_amount
                
                self.statistics['flow_augmentations'] += 1
                return True
            
            return False
        
        def relabel(u: int):
            """Relabel vertex u"""
            min_height = float('inf')
            
            for v in vertices:
                if flow[u][v] < self.graph[u][v]:
                    min_height = min(min_height, height[v])
            
            if min_height < float('inf'):
                height[u] = min_height + 1
        
        # Main preflow-push loop
        changed = True
        while changed:
            changed = False
            
            for u in vertices:
                if u != source and u != sink and excess[u] > 0:
                    # Try to push
                    pushed = False
                    for v in vertices:
                        if push(u, v):
                            pushed = True
                            changed = True
                            break
                    
                    # If couldn't push, relabel
                    if not pushed:
                        relabel(u)
                        changed = True
        
        return sum(flow[source][v] for v in vertices)
    
    def _find_augmenting_path_with_threshold(self, source: int, sink: int, 
                                           threshold: int, current_flow: Dict) -> Tuple[Optional[List[int]], int]:
        """Find augmenting path with capacity threshold"""
        parent = {}
        queue = deque([source])
        parent[source] = None
        
        while queue and sink not in parent:
            current = queue.popleft()
            
            for neighbor in self.graph[current]:
                if (neighbor not in parent and 
                    current_flow[current][neighbor] + threshold <= self.graph[current][neighbor]):
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        if sink not in parent:
            return None, 0
        
        # Reconstruct path and find bottleneck
        path = []
        bottleneck = float('inf')
        current = sink
        
        while current != source:
            path.append(current)
            prev = parent[current]
            available_capacity = self.graph[prev][current] - current_flow[prev][current]
            bottleneck = min(bottleneck, available_capacity)
            current = prev
        
        path.append(source)
        path.reverse()
        
        return path, bottleneck
    
    def _augment_flow_along_path(self, path: List[int], flow_amount: int, current_flow: Dict):
        """Augment flow along given path"""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            current_flow[u][v] += flow_amount
    
    def _find_feasible_flow(self, source: int, sink: int) -> Optional[Dict]:
        """Find a feasible flow satisfying all lower bounds"""
        # Use circulation method to find feasible flow
        circulation_graph, super_source, super_sink = self._create_circulation_network(source, sink)
        
        circulation_flow = self._solve_circulation(circulation_graph, super_source, super_sink)
        required_circulation = self._calculate_required_circulation()
        
        if circulation_flow < required_circulation:
            return None
        
        # Construct feasible flow
        feasible_flow = defaultdict(lambda: defaultdict(int))
        
        for u in self.graph:
            for v in self.graph[u]:
                feasible_flow[u][v] = self.lower_bounds[u][v]
        
        return feasible_flow
    
    def _create_residual_network_with_flow(self, flow: Dict) -> Dict:
        """Create residual network given current flow"""
        residual = defaultdict(lambda: defaultdict(int))
        
        for u in self.graph:
            for v in self.graph[u]:
                # Forward edge capacity
                residual[u][v] = self.graph[u][v] - flow[u][v]
                
                # Backward edge capacity (but respect lower bounds)
                backward_capacity = flow[u][v] - self.lower_bounds[u][v]
                if backward_capacity > 0:
                    residual[v][u] = backward_capacity
        
        return residual

def test_max_flow_with_lower_bounds():
    """Test max flow with lower bounds algorithms"""
    print("=== Testing Maximum Flow with Lower Bounds ===")
    
    # Create test network
    flow_solver = MaximumFlowWithLowerBounds()
    
    # Build test graph: (u, v, capacity, lower_bound)
    edges = [
        (0, 1, 10, 2), (0, 2, 8, 1),
        (1, 3, 5, 1), (1, 2, 2, 0),
        (2, 3, 10, 3)
    ]
    
    for u, v, capacity, lower_bound in edges:
        flow_solver.add_edge(u, v, capacity, lower_bound)
    
    source, sink = 0, 3
    
    print(f"Test network: {len(flow_solver.vertices)} vertices, {len(edges)} edges")
    print(f"Source: {source}, Sink: {sink}")
    print("Edges with (capacity, lower_bound):")
    for u, v, cap, lb in edges:
        print(f"  ({u} → {v}): capacity={cap}, lower_bound={lb}")
    
    # Test algorithms
    algorithms = [
        ("Circulation Method", flow_solver.max_flow_with_lower_bounds_circulation),
        ("Direct Transformation", flow_solver.max_flow_with_lower_bounds_direct),
        ("Preflow Push", flow_solver.max_flow_with_lower_bounds_preflow_push),
        ("Capacity Scaling", flow_solver.max_flow_with_lower_bounds_scaling),
        ("Flow Decomposition", flow_solver.max_flow_with_lower_bounds_decomposition),
    ]
    
    print(f"\nAlgorithm Results:")
    print(f"{'Algorithm':<20} | {'Feasible':<8} | {'Max Flow':<8} | {'Iterations':<10}")
    print("-" * 65)
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(source, sink)
            feasible = result['feasible']
            max_flow = result['max_flow'] if feasible else 0
            iterations = result['statistics'].get('max_flow_iterations', 0)
            
            print(f"{alg_name:<20} | {feasible:<8} | {max_flow:<8} | {iterations:<10}")
            
        except Exception as e:
            print(f"{alg_name:<20} | ERROR: {str(e)[:30]}")

def demonstrate_lower_bounds_theory():
    """Demonstrate theory of flow with lower bounds"""
    print("\n=== Flow with Lower Bounds Theory ===")
    
    print("Problem Definition:")
    print("• Standard max flow with additional lower bound constraints")
    print("• Each edge (u,v) must carry at least l(u,v) flow")
    print("• Goal: find maximum feasible flow from source to sink")
    print("• Applications: resource allocation with minimum requirements")
    
    print("\nKey Challenges:")
    print("• Feasibility: lower bounds may make problem infeasible")
    print("• Circulation: need to satisfy flow conservation with constraints")
    print("• Complexity: same as standard max flow if feasible")
    
    print("\nSolution Approaches:")
    print("1. Circulation reduction: transform to standard max flow")
    print("2. Direct transformation: adjust capacities and add demands")
    print("3. Preflow-push adaptation: initialize with lower bound flows")
    print("4. Decomposition: feasible flow + additional flow")
    
    print("\nTheoretical Results:")
    print("• Feasibility can be checked in polynomial time")
    print("• If feasible, max flow can be found in polynomial time")
    print("• Reduction preserves optimality and complexity bounds")
    print("• Applications in network design and resource planning")

if __name__ == "__main__":
    test_max_flow_with_lower_bounds()
    demonstrate_lower_bounds_theory()

"""
Maximum Flow with Lower Bounds - Key Insights:

1. **Problem Extension:**
   - Classical max flow with minimum flow requirements
   - Each edge must carry at least specified lower bound
   - Feasibility becomes a key concern
   - Applications in resource allocation with guarantees

2. **Solution Strategies:**
   - Circulation reduction: Transform to standard problem
   - Direct transformation: Adjust network structure
   - Algorithm adaptation: Modify existing algorithms
   - Decomposition: Separate feasible and additional flow

3. **Key Techniques:**
   - Super source/sink for demand satisfaction
   - Residual network construction with bounds
   - Feasibility checking via circulation
   - Flow conservation with constraints

4. **Complexity Analysis:**
   - Same asymptotic complexity as standard max flow
   - Additional feasibility checking overhead
   - Polynomial time algorithms available
   - Practical efficiency depends on network structure

5. **Applications:**
   - Network design with service guarantees
   - Resource allocation with minimum requirements
   - Transportation with capacity commitments
   - Supply chain with contract obligations

Flow with lower bounds extends classical theory
to handle practical constraints in network optimization.
"""
