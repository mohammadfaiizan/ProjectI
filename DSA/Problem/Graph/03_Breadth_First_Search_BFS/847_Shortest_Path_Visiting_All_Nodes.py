"""
847. Shortest Path Visiting All Nodes
Difficulty: Hard

Problem:
You have an undirected, connected graph of n nodes labeled from 0 to n - 1. You are 
given an array graph where graph[i] is a list of all the nodes connected to node i 
(i.e., there is an edge between node i and each node in graph[i]).

Return the length of the shortest path that visits every node. You may start and 
finish at any node.

Examples:
Input: graph = [[1,2,3],[0],[0],[0]]
Output: 4

Input: graph = [[1],[0,2,4],[1,3,4],[2],[1,2]]
Output: 4

Constraints:
- n == graph.length
- 1 <= n <= 12
- 0 <= graph[i].length < n
- graph[i] does not contain i
- graph[i] does not contain duplicate values
- The graph is connected and undirected
"""

from typing import List
from collections import deque

class Solution:
    def shortestPathLength_approach1_bitmask_bfs(self, graph: List[List[int]]) -> int:
        """
        Approach 1: Bitmask BFS (Optimal)
        
        Use BFS with state compression using bitmask to track visited nodes.
        State: (current_node, visited_mask)
        
        Time: O(N^2 * 2^N) - N^2 states * 2^N masks
        Space: O(N * 2^N) - visited states
        """
        n = len(graph)
        
        # Special case: single node
        if n == 1:
            return 0
        
        # Target: all nodes visited (all bits set)
        target_mask = (1 << n) - 1
        
        # BFS setup: (node, visited_mask, distance)
        queue = deque()
        visited = set()
        
        # Start from every node (optimization for shortest path)
        for i in range(n):
            initial_mask = 1 << i
            queue.append((i, initial_mask, 0))
            visited.add((i, initial_mask))
        
        while queue:
            node, mask, dist = queue.popleft()
            
            # Check if all nodes visited
            if mask == target_mask:
                return dist
            
            # Explore neighbors
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                state = (neighbor, new_mask)
                
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, new_mask, dist + 1))
        
        return -1  # Should not reach here for connected graph
    
    def shortestPathLength_approach2_dp_bitmask(self, graph: List[List[int]]) -> int:
        """
        Approach 2: Dynamic Programming with Bitmask
        
        Use DP to find shortest path visiting all nodes.
        dp[mask][node] = shortest path to reach 'node' visiting nodes in 'mask'
        
        Time: O(N^2 * 2^N)
        Space: O(N * 2^N)
        """
        n = len(graph)
        
        if n == 1:
            return 0
        
        # dp[mask][node] = minimum cost to visit nodes in mask ending at node
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Base case: start from each node
        for i in range(n):
            dp[1 << i][i] = 0
        
        # Fill DP table
        for mask in range(1 << n):
            for node in range(n):
                if not (mask & (1 << node)):  # node not in current mask
                    continue
                
                if dp[mask][node] == float('inf'):
                    continue
                
                # Try to extend to neighbors
                for neighbor in graph[node]:
                    new_mask = mask | (1 << neighbor)
                    dp[new_mask][neighbor] = min(dp[new_mask][neighbor], 
                                                dp[mask][node] + 1)
        
        # Find minimum cost to visit all nodes
        target_mask = (1 << n) - 1
        return min(dp[target_mask])
    
    def shortestPathLength_approach3_tsp_approximation(self, graph: List[List[int]]) -> int:
        """
        Approach 3: TSP-style Approach with Optimization
        
        Treat as Traveling Salesman Problem variant with optimizations.
        
        Time: O(N^2 * 2^N)
        Space: O(N * 2^N)
        """
        n = len(graph)
        
        if n == 1:
            return 0
        
        # Precompute shortest distances between all pairs
        dist = [[float('inf')] * n for _ in range(n)]
        
        # Initialize distances
        for i in range(n):
            dist[i][i] = 0
            for neighbor in graph[i]:
                dist[i][neighbor] = 1
        
        # Floyd-Warshall for all-pairs shortest path
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # DP with bitmask
        dp = {}
        
        def solve(mask, pos):
            """Find minimum cost to visit remaining nodes starting from pos"""
            if mask == (1 << n) - 1:
                return 0
            
            if (mask, pos) in dp:
                return dp[(mask, pos)]
            
            result = float('inf')
            
            for next_node in range(n):
                if not (mask & (1 << next_node)):  # Not visited yet
                    new_mask = mask | (1 << next_node)
                    cost = dist[pos][next_node] + solve(new_mask, next_node)
                    result = min(result, cost)
            
            dp[(mask, pos)] = result
            return result
        
        # Try starting from each node
        min_cost = float('inf')
        for start in range(n):
            cost = solve(1 << start, start)
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    def shortestPathLength_approach4_optimized_bfs(self, graph: List[List[int]]) -> int:
        """
        Approach 4: Optimized BFS with Pruning
        
        Add optimizations like early termination and state pruning.
        
        Time: O(N^2 * 2^N)
        Space: O(N * 2^N)
        """
        n = len(graph)
        
        if n == 1:
            return 0
        
        target_mask = (1 << n) - 1
        
        # Use array instead of set for better performance
        visited = {}
        queue = deque()
        
        # Multi-source BFS start
        for i in range(n):
            initial_mask = 1 << i
            queue.append((i, initial_mask, 0))
            visited[(i, initial_mask)] = 0
        
        while queue:
            node, mask, dist = queue.popleft()
            
            # Early termination
            if mask == target_mask:
                return dist
            
            # Pruning: skip if we've seen better path to this state
            if visited.get((node, mask), float('inf')) < dist:
                continue
            
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                new_dist = dist + 1
                state_key = (neighbor, new_mask)
                
                if (state_key not in visited or 
                    visited[state_key] > new_dist):
                    visited[state_key] = new_dist
                    queue.append((neighbor, new_mask, new_dist))
        
        return -1

def test_shortest_path_visiting_all_nodes():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (graph, expected)
        ([[1,2,3],[0],[0],[0]], 4),  # Star graph
        ([[1],[0,2,4],[1,3,4],[2],[1,2]], 4),  # Complex graph
        ([[]], 0),  # Single node
        ([[1],[0]], 1),  # Two nodes
        ([[1,2],[0,2],[0,1]], 2),  # Triangle
        ([[2,3],[2,3],[0,1],[0,1]], 3),  # Diamond
        ([[1,2,3,4],[0,2,6],[0,1,4],[0,4],[0,1,2,3,5],[4],[1]], 7),  # Complex connected
    ]
    
    approaches = [
        ("Bitmask BFS", solution.shortestPathLength_approach1_bitmask_bfs),
        ("DP Bitmask", solution.shortestPathLength_approach2_dp_bitmask),
        ("TSP Approximation", solution.shortestPathLength_approach3_tsp_approximation),
        ("Optimized BFS", solution.shortestPathLength_approach4_optimized_bfs),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (graph, expected) in enumerate(test_cases):
            result = func(graph)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_bitmask_state_space():
    """Demonstrate bitmask state space exploration"""
    print("\n=== Bitmask State Space Demo ===")
    
    graph = [[1,2,3],[0],[0],[0]]  # Star graph
    n = len(graph)
    
    print("Graph (star configuration):")
    print(f"  Node 0 connects to: {graph[0]}")
    for i in range(1, n):
        print(f"  Node {i} connects to: {graph[i]}")
    
    print(f"\nBitmask representation for {n} nodes:")
    for mask in range(1 << n):
        visited_nodes = []
        for i in range(n):
            if mask & (1 << i):
                visited_nodes.append(i)
        print(f"  Mask {mask:04b} ({mask:2d}): visited nodes {visited_nodes}")
    
    target_mask = (1 << n) - 1
    print(f"\nTarget mask: {target_mask:04b} (all nodes visited)")
    
    # Simulate BFS exploration
    queue = deque()
    visited = set()
    
    print(f"\nBFS exploration:")
    
    # Start from all nodes
    for i in range(n):
        initial_mask = 1 << i
        queue.append((i, initial_mask, 0))
        visited.add((i, initial_mask))
        print(f"  Start: node {i}, mask {initial_mask:04b}, distance 0")
    
    step = 0
    while queue and step < 10:  # Limit steps for demo
        step += 1
        print(f"\nStep {step}:")
        
        current_level_size = len(queue)
        for _ in range(min(current_level_size, 3)):  # Show first few
            if not queue:
                break
                
            node, mask, dist = queue.popleft()
            
            if mask == target_mask:
                print(f"  🎯 Solution found! Distance: {dist}")
                return
            
            print(f"  Process: node {node}, mask {mask:04b}, dist {dist}")
            
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                state = (neighbor, new_mask)
                
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, new_mask, dist + 1))
                    print(f"    → Add: node {neighbor}, mask {new_mask:04b}, dist {dist+1}")

def analyze_hamiltonian_path_vs_tsp():
    """Analyze relationship to Hamiltonian Path and TSP"""
    print("\n=== Hamiltonian Path vs TSP Analysis ===")
    
    print("Problem Classification:")
    print("📍 This problem: Visit all nodes (any start/end)")
    print("📍 Hamiltonian Path: Visit all nodes exactly once")
    print("📍 TSP: Visit all nodes and return to start")
    
    print("\nKey Differences:")
    print("1. **Repetition Allowed:**")
    print("   • This problem: Can revisit nodes")
    print("   • Hamiltonian: Each node exactly once")
    print("   • TSP: Each node exactly once + return")
    
    print("\n2. **Objective:**")
    print("   • This problem: Minimum edges to visit all")
    print("   • Hamiltonian: Find if path exists")
    print("   • TSP: Minimum cost tour")
    
    print("\n3. **Complexity:**")
    print("   • This problem: O(N^2 * 2^N) - tractable for N≤12")
    print("   • Hamiltonian: NP-Complete")
    print("   • TSP: NP-Hard")
    
    print("\nWhy Bitmask DP Works:")
    print("• State space: (current_node, visited_set)")
    print("• Visited set represented as bitmask")
    print("• 2^N possible visited sets")
    print("• N possible current nodes")
    print("• Total states: N * 2^N")
    
    print("\nOptimization Techniques:")
    print("• Multi-source start: Begin from all nodes")
    print("• Early termination: Stop when all visited")
    print("• State pruning: Skip worse paths to same state")
    print("• Memory optimization: Use arrays vs hash maps")
    
    print("\nReal-world Applications:")
    print("• Network maintenance: Visit all servers")
    print("• Tour planning: See all attractions")
    print("• Robot navigation: Clean all rooms")
    print("• Graph exploration: Discover all nodes")
    print("• Social networks: Reach all users")

def visualize_state_transitions():
    """Visualize state transitions in small graph"""
    print("\n=== State Transition Visualization ===")
    
    # Simple triangle graph
    graph = [[1,2],[0,2],[0,1]]
    n = len(graph)
    
    print("Graph: Triangle (0-1-2-0)")
    print("  0 -- 1")
    print("  |    |")
    print("  2 ---+")
    
    print(f"\nState space exploration:")
    
    # Manual BFS simulation for visualization
    target_mask = (1 << n) - 1
    queue = deque()
    visited = set()
    
    # Initialize
    for i in range(n):
        initial_mask = 1 << i
        queue.append((i, initial_mask, 0, [i]))
        visited.add((i, initial_mask))
    
    level = 0
    while queue and level < 4:
        level += 1
        print(f"\nLevel {level}:")
        
        level_size = len(queue)
        for _ in range(level_size):
            node, mask, dist, path = queue.popleft()
            
            visited_nodes = [i for i in range(n) if mask & (1 << i)]
            print(f"  State: node={node}, visited={visited_nodes}, dist={dist}, path={path}")
            
            if mask == target_mask:
                print(f"  🎯 Complete tour found! Distance: {dist}")
                continue
            
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                state = (neighbor, new_mask)
                
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, new_mask, dist + 1, path + [neighbor]))

if __name__ == "__main__":
    test_shortest_path_visiting_all_nodes()
    demonstrate_bitmask_state_space()
    analyze_hamiltonian_path_vs_tsp()
    visualize_state_transitions()

"""
Graph Theory Concepts:
1. Traveling Salesman Problem (TSP) Variant
2. Bitmask Dynamic Programming
3. State Space Search with Compression
4. Hamiltonian Path Approximation

Key Bitmask DP Insights:
- State: (current_node, visited_nodes_bitmask)
- Bitmask compresses visited set into integer
- BFS explores states level by level
- Multi-source start optimizes solution finding

Algorithm Strategy:
- Use bitmask to represent visited nodes
- BFS with state (node, mask, distance)
- Start from all nodes for optimization
- Return distance when all nodes visited

State Space Optimization:
- Total states: N * 2^N (manageable for N ≤ 12)
- Pruning: Skip worse paths to same state
- Early termination: Stop when target reached
- Memory efficiency: Arrays vs hash maps

Real-world Applications:
- Network maintenance routing
- Tourist itinerary optimization
- Robot exploration and mapping
- Social network analysis
- Distributed system monitoring
- Supply chain optimization

This problem demonstrates advanced state compression
techniques for exponential search spaces.
"""
