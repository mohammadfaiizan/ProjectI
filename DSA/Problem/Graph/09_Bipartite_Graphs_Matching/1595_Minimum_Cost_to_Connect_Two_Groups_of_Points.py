"""
1595. Minimum Cost to Connect Two Groups of Points - Multiple Approaches
Difficulty: Hard

You are given two groups of points where the first group has size1 points, the second group has size2 points, and size1 >= size2.

The cost of the connection between any two points are given in an m x n matrix where cost[i][j] is the cost of connecting point i of the first group and point j of the second group. The groups are connected if each point in both groups is connected to one or more points in the opposite group. In other words, each point in the first group must be connected to at least one point in the second group, and each point in the second group must be connected to at least one point in the first group.

Return the minimum cost to connect the two groups.
"""

from typing import List
from functools import lru_cache

class MinimumCostToConnect:
    """Multiple approaches to find minimum cost to connect two groups"""
    
    def connectTwoGroups_bitmask_dp(self, cost: List[List[int]]) -> int:
        """
        Approach 1: Bitmask Dynamic Programming
        
        Use bitmask to represent connected points in group 2.
        
        Time: O(m * 2^n * n), Space: O(m * 2^n)
        """
        m, n = len(cost), len(cost[0])
        
        # Precompute minimum cost to connect each point in group 2
        min_cost_group2 = [min(cost[i][j] for i in range(m)) for j in range(n)]
        
        @lru_cache(maxsize=None)
        def dp(i: int, mask: int) -> int:
            """
            dp(i, mask) = minimum cost to connect first i points of group 1
            such that points in group 2 represented by mask are connected
            """
            if i == m:
                # Connect remaining unconnected points in group 2
                remaining_cost = 0
                for j in range(n):
                    if not (mask & (1 << j)):
                        remaining_cost += min_cost_group2[j]
                return remaining_cost
            
            # Try all possible connections for point i in group 1
            min_cost = float('inf')
            
            # Try connecting point i to different subsets of group 2
            for subset in range(1, 1 << n):
                connection_cost = 0
                new_mask = mask
                
                # Calculate cost of connecting point i to this subset
                for j in range(n):
                    if subset & (1 << j):
                        connection_cost += cost[i][j]
                        new_mask |= (1 << j)
                
                total_cost = connection_cost + dp(i + 1, new_mask)
                min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        return dp(0, 0)
    
    def connectTwoGroups_optimized_dp(self, cost: List[List[int]]) -> int:
        """
        Approach 2: Optimized DP with Better State Management
        
        Optimize the DP transition and state representation.
        
        Time: O(m * 2^n * n), Space: O(2^n)
        """
        m, n = len(cost), len(cost[0])
        
        # Precompute minimum costs
        min_cost_to_connect_j = [min(cost[i][j] for i in range(m)) for j in range(n)]
        
        # DP table: dp[mask] = minimum cost when group 2 points in mask are connected
        prev_dp = [float('inf')] * (1 << n)
        prev_dp[0] = 0
        
        for i in range(m):
            curr_dp = [float('inf')] * (1 << n)
            
            for mask in range(1 << n):
                if prev_dp[mask] == float('inf'):
                    continue
                
                # Try all possible ways to connect point i
                for new_connections in range(1, 1 << n):
                    connection_cost = 0
                    new_mask = mask
                    
                    for j in range(n):
                        if new_connections & (1 << j):
                            connection_cost += cost[i][j]
                            new_mask |= (1 << j)
                    
                    curr_dp[new_mask] = min(curr_dp[new_mask], 
                                          prev_dp[mask] + connection_cost)
            
            prev_dp = curr_dp
        
        # Add cost for unconnected points in group 2
        result = float('inf')
        for mask in range(1 << n):
            if prev_dp[mask] == float('inf'):
                continue
            
            additional_cost = 0
            for j in range(n):
                if not (mask & (1 << j)):
                    additional_cost += min_cost_to_connect_j[j]
            
            result = min(result, prev_dp[mask] + additional_cost)
        
        return result
    
    def connectTwoGroups_min_cost_flow(self, cost: List[List[int]]) -> int:
        """
        Approach 3: Minimum Cost Flow Modeling
        
        Model as minimum cost flow problem with capacity constraints.
        
        Time: O(V²E), Space: O(V + E)
        """
        m, n = len(cost), len(cost[0])
        
        # Create flow network
        # Nodes: source(0), group1(1..m), group2(m+1..m+n), sink(m+n+1)
        total_nodes = m + n + 2
        source, sink = 0, total_nodes - 1
        
        # Adjacency list with (neighbor, capacity, cost)
        graph = [[] for _ in range(total_nodes)]
        
        # Source to group 1 (each point must be connected at least once)
        for i in range(1, m + 1):
            graph[source].append((i, 1, 0))
            graph[i].append((source, 0, 0))  # Reverse edge
        
        # Group 1 to group 2 (connection costs)
        for i in range(1, m + 1):
            for j in range(m + 1, m + n + 1):
                group1_idx = i - 1
                group2_idx = j - m - 1
                graph[i].append((j, 1, cost[group1_idx][group2_idx]))
                graph[j].append((i, 0, -cost[group1_idx][group2_idx]))  # Reverse edge
        
        # Group 2 to sink (each point must be connected at least once)
        for j in range(m + 1, m + n + 1):
            graph[j].append((sink, 1, 0))
            graph[sink].append((j, 0, 0))  # Reverse edge
        
        # Additional edges to ensure all points are connected
        # Add high capacity edges from source to group 2 with minimum costs
        for j in range(m + 1, m + n + 1):
            group2_idx = j - m - 1
            min_cost_to_j = min(cost[i][group2_idx] for i in range(m))
            graph[source].append((j, m, min_cost_to_j))
            graph[j].append((source, 0, -min_cost_to_j))
        
        # Simplified min cost flow (for small instances)
        return self._simplified_min_cost_flow(graph, source, sink, m + n)
    
    def _simplified_min_cost_flow(self, graph: List[List], source: int, sink: int, required_flow: int) -> int:
        """Simplified min cost flow implementation"""
        # For simplicity, use a basic approach
        # In practice, would use more sophisticated algorithms like Successive Shortest Path
        
        total_cost = 0
        flow_sent = 0
        
        while flow_sent < required_flow:
            # Find shortest path using Bellman-Ford (handles negative weights)
            dist = [float('inf')] * len(graph)
            parent = [-1] * len(graph)
            parent_edge_idx = [-1] * len(graph)
            
            dist[source] = 0
            
            # Relax edges
            for _ in range(len(graph)):
                updated = False
                for u in range(len(graph)):
                    if dist[u] == float('inf'):
                        continue
                    
                    for edge_idx, (v, capacity, cost) in enumerate(graph[u]):
                        if capacity > 0 and dist[u] + cost < dist[v]:
                            dist[v] = dist[u] + cost
                            parent[v] = u
                            parent_edge_idx[v] = edge_idx
                            updated = True
                
                if not updated:
                    break
            
            if dist[sink] == float('inf'):
                break  # No more augmenting paths
            
            # Find bottleneck capacity
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                edge_idx = parent_edge_idx[v]
                capacity = graph[u][edge_idx][1]
                path_flow = min(path_flow, capacity)
                v = u
            
            # Update residual capacities and cost
            v = sink
            while v != source:
                u = parent[v]
                edge_idx = parent_edge_idx[v]
                
                # Forward edge
                graph[u][edge_idx] = (graph[u][edge_idx][0], 
                                    graph[u][edge_idx][1] - path_flow,
                                    graph[u][edge_idx][2])
                
                # Find and update reverse edge
                for i, (neighbor, cap, c) in enumerate(graph[v]):
                    if neighbor == u:
                        graph[v][i] = (neighbor, cap + path_flow, c)
                        break
                
                v = u
            
            total_cost += path_flow * dist[sink]
            flow_sent += path_flow
        
        return total_cost if flow_sent >= required_flow else float('inf')
    
    def connectTwoGroups_greedy_with_correction(self, cost: List[List[int]]) -> int:
        """
        Approach 4: Greedy Algorithm with Correction
        
        Start with greedy assignment then correct for constraints.
        
        Time: O(m * n * 2^n), Space: O(2^n)
        """
        m, n = len(cost), len(cost[0])
        
        # Find minimum cost assignment (Hungarian-style)
        def min_assignment_cost():
            # For each point in group 2, find minimum cost connection
            min_costs = []
            for j in range(n):
                min_cost = min(cost[i][j] for i in range(m))
                min_costs.append(min_cost)
            return sum(min_costs)
        
        base_cost = min_assignment_cost()
        
        # Now ensure all points in group 1 are connected
        # Use DP to find minimum additional cost
        @lru_cache(maxsize=None)
        def dp(i: int, connected_mask: int) -> int:
            if i == m:
                return 0
            
            min_cost = float('inf')
            
            # Point i must connect to at least one point in group 2
            for j in range(n):
                new_mask = connected_mask | (1 << j)
                additional_cost = 0
                
                # If this connection is not in the base assignment, add its cost
                if not (connected_mask & (1 << j)):
                    additional_cost = cost[i][j] - min(cost[k][j] for k in range(m))
                
                total_cost = additional_cost + dp(i + 1, new_mask)
                min_cost = min(min_cost, total_cost)
            
            return min_cost
        
        additional_cost = dp(0, 0)
        return base_cost + additional_cost
    
    def connectTwoGroups_branch_and_bound(self, cost: List[List[int]]) -> int:
        """
        Approach 5: Branch and Bound
        
        Use branch and bound with intelligent pruning.
        
        Time: O(exponential), Space: O(m + n)
        """
        m, n = len(cost), len(cost[0])
        
        # Precompute bounds
        min_cost_group1 = [min(cost[i]) for i in range(m)]
        min_cost_group2 = [min(cost[i][j] for i in range(m)) for j in range(n)]
        
        self.best_cost = float('inf')
        
        def lower_bound(i: int, connected_group2: set) -> int:
            """Calculate lower bound for remaining connections"""
            bound = 0
            
            # Remaining points in group 1
            for k in range(i, m):
                bound += min_cost_group1[k]
            
            # Unconnected points in group 2
            for j in range(n):
                if j not in connected_group2:
                    bound += min_cost_group2[j]
            
            return bound
        
        def branch_and_bound(i: int, current_cost: int, connected_group2: set):
            if current_cost >= self.best_cost:
                return  # Pruning
            
            if i == m:
                # Add cost for unconnected points in group 2
                additional_cost = sum(min_cost_group2[j] for j in range(n) 
                                    if j not in connected_group2)
                self.best_cost = min(self.best_cost, current_cost + additional_cost)
                return
            
            # Pruning with lower bound
            if current_cost + lower_bound(i, connected_group2) >= self.best_cost:
                return
            
            # Try connecting point i to different points in group 2
            for j in range(n):
                new_connected = connected_group2 | {j}
                new_cost = current_cost + cost[i][j]
                branch_and_bound(i + 1, new_cost, new_connected)
        
        branch_and_bound(0, 0, set())
        return self.best_cost

def test_minimum_cost_to_connect():
    """Test minimum cost to connect two groups algorithms"""
    solver = MinimumCostToConnect()
    
    test_cases = [
        ([[15, 96], [36, 2]], 17, "Example 1"),
        ([[1, 3, 5], [4, 1, 1], [1, 5, 3]], 4, "Example 2"),
        ([[2, 5, 1], [3, 4, 7], [8, 1, 2], [6, 2, 4], [3, 8, 8]], 10, "Example 3"),
        ([[1, 2], [3, 4]], 4, "Simple 2x2"),
    ]
    
    algorithms = [
        ("Bitmask DP", solver.connectTwoGroups_bitmask_dp),
        ("Optimized DP", solver.connectTwoGroups_optimized_dp),
        ("Greedy with Correction", solver.connectTwoGroups_greedy_with_correction),
        ("Branch and Bound", solver.connectTwoGroups_branch_and_bound),
    ]
    
    print("=== Testing Minimum Cost to Connect Two Groups ===")
    
    for cost, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Cost matrix: {cost}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(cost)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:22} | {status} | Cost: {result}")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_minimum_cost_to_connect()

"""
Minimum Cost to Connect Two Groups demonstrates advanced
bipartite matching with constraints, dynamic programming
with bitmasks, and minimum cost flow modeling.
"""
