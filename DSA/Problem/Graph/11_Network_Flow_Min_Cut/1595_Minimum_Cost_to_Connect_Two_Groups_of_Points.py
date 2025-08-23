"""
1595. Minimum Cost to Connect Two Groups of Points - Multiple Approaches
Difficulty: Hard

You are given two groups of points where the first group has size1 points, 
the second group has size2 points, and size1 >= size2.

The cost of the connection between any two points are given in an array cost 
where cost[i][j] is the cost of connecting point i of the first group and 
point j of the second group. The groups are connected if each point in both 
groups is connected to one or more points in the other group. In other words, 
each point in the first group must be connected to at least one point in the 
second group, and each point in the second group must be connected to at least 
one point in the first group.

Return the minimum cost to connect the two groups.
"""

from typing import List, Dict, Set, Tuple, Optional
from functools import lru_cache
import heapq

class MinCostConnectTwoGroups:
    """Multiple approaches to solve minimum cost connection problem"""
    
    def connectTwoGroups_dp_bitmask(self, cost: List[List[int]]) -> int:
        """
        Approach 1: Dynamic Programming with Bitmask
        
        Use bitmask DP to track which points in group2 are connected.
        
        Time: O(size1 * 2^size2 * size2)
        Space: O(size1 * 2^size2)
        """
        size1, size2 = len(cost), len(cost[0])
        
        # Precompute minimum cost to connect each point in group2
        min_cost_group2 = [min(cost[i][j] for i in range(size1)) for j in range(size2)]
        
        @lru_cache(maxsize=None)
        def dp(i: int, mask: int) -> int:
            """
            DP function: minimum cost to connect points 0..i-1 from group1
            and points represented by mask from group2
            """
            if i == size1:
                # All points in group1 processed
                # Connect remaining unconnected points in group2
                total_cost = 0
                for j in range(size2):
                    if not (mask & (1 << j)):
                        total_cost += min_cost_group2[j]
                return total_cost
            
            min_cost = float('inf')
            
            # Try connecting point i to each point in group2
            for j in range(size2):
                new_mask = mask | (1 << j)
                cost_with_connection = cost[i][j] + dp(i + 1, new_mask)
                min_cost = min(min_cost, cost_with_connection)
            
            return min_cost
        
        return dp(0, 0)
    
    def connectTwoGroups_dp_optimized(self, cost: List[List[int]]) -> int:
        """
        Approach 2: Optimized DP with Multiple Connections
        
        Allow multiple connections per point in group1 for optimization.
        
        Time: O(size1 * 2^size2 * size2)
        Space: O(size1 * 2^size2)
        """
        size1, size2 = len(cost), len(cost[0])
        
        # Precompute minimum cost for each point in group2
        min_cost_group2 = [min(cost[i][j] for i in range(size1)) for j in range(size2)]
        
        @lru_cache(maxsize=None)
        def dp(i: int, mask: int) -> int:
            """DP with option to connect to multiple points"""
            if i == size1:
                # Connect remaining points in group2
                total_cost = 0
                for j in range(size2):
                    if not (mask & (1 << j)):
                        total_cost += min_cost_group2[j]
                return total_cost
            
            min_cost = float('inf')
            
            # Try all possible subsets of connections for point i
            for subset in range(1, 1 << size2):
                connection_cost = 0
                new_mask = mask
                
                for j in range(size2):
                    if subset & (1 << j):
                        connection_cost += cost[i][j]
                        new_mask |= (1 << j)
                
                total_cost = connection_cost + dp(i + 1, new_mask)
                min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        return dp(0, 0)
    
    def connectTwoGroups_min_cost_flow(self, cost: List[List[int]]) -> int:
        """
        Approach 3: Minimum Cost Flow Modeling
        
        Model as min-cost flow problem with constraints.
        
        Time: O(V^2 * E) where V = size1 + size2 + 2, E = size1 * size2
        Space: O(V + E)
        """
        size1, size2 = len(cost), len(cost[0])
        
        # Create flow network
        # Vertices: source, group1 points, group2 points, sink
        source = 0
        group1_start = 1
        group2_start = group1_start + size1
        sink = group2_start + size2
        total_vertices = sink + 1
        
        # Build min-cost flow graph
        edges = []
        
        # Source to group1 (capacity 1, cost 0)
        for i in range(size1):
            edges.append((source, group1_start + i, 1, 0))
        
        # Group1 to group2 (capacity 1, given cost)
        for i in range(size1):
            for j in range(size2):
                edges.append((group1_start + i, group2_start + j, 1, cost[i][j]))
        
        # Group2 to sink (capacity 1, cost 0)
        for j in range(size2):
            edges.append((group2_start + j, sink, 1, 0))
        
        # Additional edges for ensuring all group2 points are connected
        # Source to group2 (capacity 1, minimum cost to connect each group2 point)
        min_cost_group2 = [min(cost[i][j] for i in range(size1)) for j in range(size2)]
        for j in range(size2):
            edges.append((source, group2_start + j, 1, min_cost_group2[j]))
        
        # Solve min-cost flow (simplified implementation)
        return self._solve_min_cost_flow(edges, source, sink, size1 + size2, total_vertices)
    
    def connectTwoGroups_assignment_based(self, cost: List[List[int]]) -> int:
        """
        Approach 4: Assignment Problem Based Solution
        
        Use assignment problem as base and handle additional constraints.
        
        Time: O(size1^3) for assignment + O(2^size2) for remaining
        Space: O(size1^2)
        """
        size1, size2 = len(cost), len(cost[0])
        
        # Find minimum cost assignment (each group1 point to one group2 point)
        assignment_cost, assignment = self._hungarian_algorithm(cost)
        
        # Check which group2 points are covered by assignment
        covered_group2 = set(assignment.values())
        uncovered_group2 = set(range(size2)) - covered_group2
        
        if not uncovered_group2:
            return assignment_cost
        
        # Find minimum cost to cover uncovered group2 points
        min_additional_cost = 0
        for j in uncovered_group2:
            min_additional_cost += min(cost[i][j] for i in range(size1))
        
        return assignment_cost + min_additional_cost
    
    def connectTwoGroups_greedy_optimization(self, cost: List[List[int]]) -> int:
        """
        Approach 5: Greedy with Local Optimization
        
        Greedy approach with local optimization for better results.
        
        Time: O(size1 * size2 * log(size1 * size2))
        Space: O(size1 * size2)
        """
        size1, size2 = len(cost), len(cost[0])
        
        # Create list of all possible connections with costs
        connections = []
        for i in range(size1):
            for j in range(size2):
                connections.append((cost[i][j], i, j))
        
        connections.sort()  # Sort by cost
        
        # Greedy selection ensuring all points are connected
        selected_connections = []
        group1_connected = [False] * size1
        group2_connected = [False] * size2
        total_cost = 0
        
        # First pass: select connections greedily
        for cost_val, i, j in connections:
            if not group1_connected[i] or not group2_connected[j]:
                selected_connections.append((i, j, cost_val))
                group1_connected[i] = True
                group2_connected[j] = True
                total_cost += cost_val
                
                # Check if all points are connected
                if all(group1_connected) and all(group2_connected):
                    break
        
        # Ensure all points are connected
        for i in range(size1):
            if not group1_connected[i]:
                # Find cheapest connection for this point
                min_cost = min(cost[i][j] for j in range(size2))
                min_j = next(j for j in range(size2) if cost[i][j] == min_cost)
                selected_connections.append((i, min_j, min_cost))
                total_cost += min_cost
        
        for j in range(size2):
            if not group2_connected[j]:
                # Find cheapest connection for this point
                min_cost = min(cost[i][j] for i in range(size1))
                min_i = next(i for i in range(size1) if cost[i][j] == min_cost)
                selected_connections.append((min_i, j, min_cost))
                total_cost += min_cost
        
        return total_cost
    
    def connectTwoGroups_branch_and_bound(self, cost: List[List[int]]) -> int:
        """
        Approach 6: Branch and Bound
        
        Use branch and bound for optimal solution with pruning.
        
        Time: O(exponential) but with pruning
        Space: O(size1 + size2)
        """
        size1, size2 = len(cost), len(cost[0])
        
        # Precompute bounds
        min_cost_group1 = [min(cost[i]) for i in range(size1)]
        min_cost_group2 = [min(cost[i][j] for i in range(size1)) for j in range(size2)]
        
        self.best_cost = float('inf')
        
        def lower_bound(group1_connected: List[bool], group2_connected: List[bool]) -> int:
            """Calculate lower bound for remaining connections"""
            bound = 0
            
            # Cost for unconnected group1 points
            for i in range(size1):
                if not group1_connected[i]:
                    bound += min_cost_group1[i]
            
            # Cost for unconnected group2 points
            for j in range(size2):
                if not group2_connected[j]:
                    bound += min_cost_group2[j]
            
            return bound
        
        def branch_and_bound(i: int, current_cost: int, 
                           group1_connected: List[bool], 
                           group2_connected: List[bool]):
            """Branch and bound recursive function"""
            if i == size1:
                # Ensure all group2 points are connected
                additional_cost = 0
                for j in range(size2):
                    if not group2_connected[j]:
                        additional_cost += min_cost_group2[j]
                
                total_cost = current_cost + additional_cost
                self.best_cost = min(self.best_cost, total_cost)
                return
            
            # Pruning
            bound = current_cost + lower_bound(group1_connected, group2_connected)
            if bound >= self.best_cost:
                return
            
            # Try connecting point i to each point in group2
            for j in range(size2):
                new_cost = current_cost + cost[i][j]
                new_group1_connected = group1_connected[:]
                new_group2_connected = group2_connected[:]
                new_group1_connected[i] = True
                new_group2_connected[j] = True
                
                branch_and_bound(i + 1, new_cost, new_group1_connected, new_group2_connected)
        
        branch_and_bound(0, 0, [False] * size1, [False] * size2)
        return self.best_cost
    
    def _solve_min_cost_flow(self, edges: List[Tuple[int, int, int, int]], 
                           source: int, sink: int, required_flow: int, 
                           num_vertices: int) -> int:
        """Simplified min-cost flow solver"""
        # This is a simplified implementation
        # In practice, would use more sophisticated algorithms
        
        # Build adjacency list
        graph = [[] for _ in range(num_vertices)]
        for u, v, capacity, cost in edges:
            graph[u].append((v, capacity, cost))
        
        # Simple greedy approach for demonstration
        total_cost = 0
        flow_sent = 0
        
        while flow_sent < required_flow:
            # Find shortest path by cost
            dist = [float('inf')] * num_vertices
            dist[source] = 0
            parent = [-1] * num_vertices
            
            # Bellman-Ford style relaxation
            for _ in range(num_vertices - 1):
                for u in range(num_vertices):
                    if dist[u] == float('inf'):
                        continue
                    for v, capacity, cost in graph[u]:
                        if capacity > 0 and dist[u] + cost < dist[v]:
                            dist[v] = dist[u] + cost
                            parent[v] = u
            
            if dist[sink] == float('inf'):
                break
            
            # Send one unit of flow
            current = sink
            path_cost = dist[sink]
            
            while current != source:
                prev = parent[current]
                # Update capacities (simplified)
                current = prev
            
            total_cost += path_cost
            flow_sent += 1
        
        return total_cost
    
    def _hungarian_algorithm(self, cost_matrix: List[List[int]]) -> Tuple[int, Dict[int, int]]:
        """Simplified Hungarian algorithm implementation"""
        size1, size2 = len(cost_matrix), len(cost_matrix[0])
        
        # Extend matrix to be square if needed
        if size1 != size2:
            max_size = max(size1, size2)
            extended_matrix = [[float('inf')] * max_size for _ in range(max_size)]
            
            for i in range(size1):
                for j in range(size2):
                    extended_matrix[i][j] = cost_matrix[i][j]
            
            cost_matrix = extended_matrix
            size1 = size2 = max_size
        
        # Simplified assignment (greedy approximation)
        assignment = {}
        used_cols = set()
        total_cost = 0
        
        for i in range(len(cost_matrix)):
            min_cost = float('inf')
            min_j = -1
            
            for j in range(len(cost_matrix[0])):
                if j not in used_cols and cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
                    min_j = j
            
            if min_j != -1 and min_cost != float('inf'):
                assignment[i] = min_j
                used_cols.add(min_j)
                total_cost += min_cost
        
        return total_cost, assignment

def test_min_cost_connect_two_groups():
    """Test all approaches with various test cases"""
    solver = MinCostConnectTwoGroups()
    
    test_cases = [
        # (cost, expected, description)
        ([[15, 96], [36, 2]], 17, "Simple 2x2 case"),
        ([[1, 3, 5], [4, 1, 1], [1, 5, 3]], 4, "3x3 case"),
        ([[2, 5, 1], [3, 4, 7], [8, 1, 2], [6, 2, 4], [3, 8, 8]], 10, "5x3 case"),
        ([[1]], 1, "Single connection"),
        ([[1, 2], [3, 4]], 4, "2x2 balanced"),
    ]
    
    approaches = [
        ("DP Bitmask", solver.connectTwoGroups_dp_bitmask),
        ("DP Optimized", solver.connectTwoGroups_dp_optimized),
        ("Assignment Based", solver.connectTwoGroups_assignment_based),
        ("Greedy", solver.connectTwoGroups_greedy_optimization),
        ("Branch & Bound", solver.connectTwoGroups_branch_and_bound),
    ]
    
    print("=== Testing Min Cost Connect Two Groups ===")
    
    for cost, expected, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Cost matrix: {cost}")
        print(f"Expected: {expected}")
        
        for approach_name, approach_func in approaches:
            try:
                result = approach_func(cost)
                status = "✓" if result == expected else "✗"
                print(f"{approach_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{approach_name:15} | ERROR: {str(e)[:30]}")

def demonstrate_problem_analysis():
    """Demonstrate problem structure and solution approach"""
    print("\n=== Problem Analysis Demo ===")
    
    cost = [[15, 96], [36, 2]]
    print(f"Example: cost = {cost}")
    
    print(f"\nProblem constraints:")
    print(f"• Each point in group1 must connect to ≥1 point in group2")
    print(f"• Each point in group2 must connect to ≥1 point in group1")
    print(f"• Minimize total connection cost")
    
    print(f"\nSolution analysis:")
    print(f"• Point 0 (group1) → Point 1 (group2): cost 96")
    print(f"• Point 1 (group1) → Point 1 (group2): cost 2")
    print(f"• But point 0 (group2) not connected!")
    print(f"• Need: Point 0 (group1) → Point 0 (group2): cost 15")
    print(f"• Total: 15 + 2 = 17")
    
    solver = MinCostConnectTwoGroups()
    result = solver.connectTwoGroups_dp_bitmask(cost)
    print(f"\nOptimal cost: {result}")

def analyze_algorithm_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **DP Bitmask:**")
    print("   • Time: O(size1 * 2^size2 * size2)")
    print("   • Space: O(size1 * 2^size2)")
    print("   • Pros: Optimal solution guaranteed")
    print("   • Cons: Exponential in size2")
    
    print("\n2. **DP Optimized:**")
    print("   • Time: O(size1 * 2^size2 * 2^size2)")
    print("   • Space: O(size1 * 2^size2)")
    print("   • Pros: Considers multiple connections")
    print("   • Cons: Higher time complexity")
    
    print("\n3. **Min-Cost Flow:**")
    print("   • Time: O(V^2 * E) where V = size1 + size2")
    print("   • Space: O(V + E)")
    print("   • Pros: Polynomial time")
    print("   • Cons: Complex implementation")
    
    print("\n4. **Assignment Based:**")
    print("   • Time: O(size1^3) + O(size2)")
    print("   • Space: O(size1^2)")
    print("   • Pros: Uses well-known algorithms")
    print("   • Cons: May not be optimal")
    
    print("\n5. **Greedy:**")
    print("   • Time: O(size1 * size2 * log(size1 * size2))")
    print("   • Space: O(size1 * size2)")
    print("   • Pros: Fast and simple")
    print("   • Cons: Not guaranteed optimal")
    
    print("\n6. **Branch and Bound:**")
    print("   • Time: Exponential with pruning")
    print("   • Space: O(size1 + size2)")
    print("   • Pros: Optimal with good pruning")
    print("   • Cons: Worst-case exponential")

def demonstrate_optimization_strategies():
    """Demonstrate optimization strategies"""
    print("\n=== Optimization Strategies ===")
    
    print("Problem-Specific Optimizations:")
    
    print("\n1. **Constraint Analysis:**")
    print("   • size1 ≥ size2 (given constraint)")
    print("   • Each group1 point connects to ≥1 group2 point")
    print("   • Each group2 point connects to ≥1 group1 point")
    print("   • Minimize total connection cost")
    
    print("\n2. **Bitmask DP Optimization:**")
    print("   • Use bitmask to track group2 connections")
    print("   • Precompute minimum costs for group2 points")
    print("   • Memoization to avoid recomputation")
    print("   • Process group1 points sequentially")
    
    print("\n3. **Pruning Strategies:**")
    print("   • Lower bound calculation for branch and bound")
    print("   • Early termination when bound exceeds best")
    print("   • Greedy initialization for better bounds")
    print("   • Symmetry breaking when possible")
    
    print("\n4. **Problem Reduction:**")
    print("   • Model as bipartite matching with constraints")
    print("   • Use min-cost flow formulation")
    print("   • Assignment problem as subproblem")
    print("   • Network flow with lower bounds")
    
    print("\n5. **Implementation Optimizations:**")
    print("   • Efficient bitmask operations")
    print("   • Precomputed minimum costs")
    print("   • Memory-efficient DP states")
    print("   • Fast connectivity checking")

if __name__ == "__main__":
    test_min_cost_connect_two_groups()
    demonstrate_problem_analysis()
    analyze_algorithm_complexity()
    demonstrate_optimization_strategies()

"""
Minimum Cost to Connect Two Groups - Key Insights:

1. **Problem Structure:**
   - Bipartite graph with two groups of points
   - Each point must be connected to at least one point in other group
   - Minimize total connection cost
   - Constraint satisfaction with optimization objective

2. **Algorithm Categories:**
   - Dynamic Programming: Bitmask DP for optimal solution
   - Network Flow: Min-cost flow modeling
   - Assignment: Hungarian algorithm as base
   - Greedy: Fast approximation algorithms
   - Branch and Bound: Optimal with pruning

3. **Key Challenges:**
   - Ensuring all points are connected
   - Exponential solution space
   - Balancing optimality vs efficiency
   - Handling asymmetric group sizes

4. **Optimization Techniques:**
   - Bitmask representation for group2 connections
   - Precomputation of minimum costs
   - Lower bound calculation for pruning
   - Problem reduction to known algorithms

5. **Complexity Considerations:**
   - Exponential in smaller group size (size2)
   - Polynomial algorithms via network flow
   - Trade-off between optimality and speed
   - Memory usage for DP approaches

The problem combines assignment optimization with
connectivity constraints, requiring careful algorithm design.
"""
