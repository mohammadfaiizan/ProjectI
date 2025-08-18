"""
1192. Critical Connections in a Network
Difficulty: Hard

Problem:
There are n servers numbered from 0 to n - 1 connected by undirected server-to-server 
connections forming a network where connections[i] = [ai, bi] represents a connection 
between servers ai and bi. Any server can reach other servers directly or indirectly 
through the network.

A critical connection is a connection that, if removed, will make some servers unable 
to reach some other servers.

Return all critical connections in the network in any order.

Examples:
Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
Output: [[1,3]]

Input: n = 2, connections = [[0,1]]
Output: [[0,1]]

Constraints:
- 2 <= n <= 10^5
- n - 1 <= connections.length <= 10^5
- 0 <= ai, bi <= n - 1
- ai != bi
- There are no repeated connections
"""

from typing import List
from collections import defaultdict

class Solution:
    def criticalConnections_approach1_tarjan_algorithm(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Tarjan's Bridge-Finding Algorithm (Optimal)
        
        Use DFS with discovery time and low-link values to find bridges.
        A bridge is an edge whose removal increases the number of connected components.
        
        Time: O(V + E) - single DFS traversal
        Space: O(V + E) - adjacency list + DFS stack
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in connections:
            graph[u].append(v)
            graph[v].append(u)
        
        # Tarjan's algorithm variables
        discovery_time = [-1] * n  # Discovery time of each node
        low_link = [-1] * n        # Lowest discovery time reachable from subtree
        parent = [-1] * n          # Parent in DFS tree
        bridges = []               # Result: critical connections
        time = [0]                 # Current time (use list for reference)
        
        def tarjan_dfs(u):
            """Tarjan's DFS to find bridges"""
            # Initialize discovery time and low-link value
            discovery_time[u] = low_link[u] = time[0]
            time[0] += 1
            
            # Explore all neighbors
            for v in graph[u]:
                if discovery_time[v] == -1:  # Tree edge (unvisited)
                    parent[v] = u
                    tarjan_dfs(v)
                    
                    # Update low-link value
                    low_link[u] = min(low_link[u], low_link[v])
                    
                    # Check if edge (u,v) is a bridge
                    if low_link[v] > discovery_time[u]:
                        bridges.append([u, v])
                
                elif v != parent[u]:  # Back edge (not to parent)
                    low_link[u] = min(low_link[u], discovery_time[v])
        
        # Run DFS from each unvisited node
        for i in range(n):
            if discovery_time[i] == -1:
                tarjan_dfs(i)
        
        return bridges
    
    def criticalConnections_approach2_naive_bridge_detection(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Naive Bridge Detection (For Comparison)
        
        For each edge, remove it and check if graph becomes disconnected.
        Much slower but easier to understand.
        
        Time: O(E * (V + E)) - for each edge, run DFS
        Space: O(V + E)
        """
        def is_connected_without_edge(removed_edge):
            """Check if graph is connected after removing an edge"""
            # Build graph without the removed edge
            graph = defaultdict(list)
            for u, v in connections:
                if [u, v] != removed_edge and [v, u] != removed_edge:
                    graph[u].append(v)
                    graph[v].append(u)
            
            # DFS to check connectivity
            visited = [False] * n
            
            def dfs(node):
                visited[node] = True
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        dfs(neighbor)
            
            # Start DFS from node 0
            dfs(0)
            
            # Check if all nodes are visited
            return all(visited)
        
        bridges = []
        for edge in connections:
            if not is_connected_without_edge(edge):
                bridges.append(edge)
        
        return bridges
    
    def criticalConnections_approach3_optimized_tarjan(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Optimized Tarjan with Early Termination
        
        Add optimizations to Tarjan's algorithm for better performance.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list with edge indices for faster lookup
        graph = defaultdict(list)
        edge_set = set()
        
        for i, (u, v) in enumerate(connections):
            graph[u].append(v)
            graph[v].append(u)
            edge_set.add((min(u, v), max(u, v)))  # Normalized edge representation
        
        discovery = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        bridges = []
        time = [0]
        
        def dfs(u):
            discovery[u] = low[u] = time[0]
            time[0] += 1
            
            for v in graph[u]:
                if discovery[v] == -1:  # Tree edge
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # Bridge condition
                    if low[v] > discovery[u]:
                        # Ensure consistent edge order
                        bridge = [min(u, v), max(u, v)]
                        if tuple(bridge) in edge_set:
                            bridges.append(bridge)
                
                elif v != parent[u]:  # Back edge
                    low[u] = min(low[u], discovery[v])
        
        # Handle disconnected components
        for i in range(n):
            if discovery[i] == -1:
                dfs(i)
        
        return bridges
    
    def criticalConnections_approach4_iterative_tarjan(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: Iterative Tarjan (Avoid Recursion)
        
        Implement Tarjan's algorithm iteratively to handle large graphs.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in connections:
            graph[u].append(v)
            graph[v].append(u)
        
        discovery = [-1] * n
        low = [-1] * n
        parent = [-1] * n
        bridges = []
        time = [0]
        
        def iterative_tarjan(start):
            """Iterative implementation of Tarjan's algorithm"""
            stack = [(start, 0, iter(graph[start]))]  # (node, state, neighbors_iter)
            
            while stack:
                u, state, neighbors = stack[-1]
                
                if state == 0:  # First visit
                    discovery[u] = low[u] = time[0]
                    time[0] += 1
                    stack[-1] = (u, 1, neighbors)  # Update state
                
                try:
                    v = next(neighbors)
                    
                    if discovery[v] == -1:  # Tree edge
                        parent[v] = u
                        stack.append((v, 0, iter(graph[v])))
                    elif v != parent[u]:  # Back edge
                        low[u] = min(low[u], discovery[v])
                
                except StopIteration:  # No more neighbors
                    stack.pop()
                    
                    if stack:  # Update parent's low-link value
                        p, _, _ = stack[-1]
                        low[p] = min(low[p], low[u])
                        
                        # Check bridge condition
                        if low[u] > discovery[p]:
                            bridges.append([p, u])
        
        # Run for each connected component
        for i in range(n):
            if discovery[i] == -1:
                iterative_tarjan(i)
        
        return bridges

def test_critical_connections():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, connections, expected)
        (4, [[0,1],[1,2],[2,0],[1,3]], [[1,3]]),
        (2, [[0,1]], [[0,1]]),
        (6, [[0,1],[1,2],[2,0],[1,3],[3,4],[4,5],[5,3]], [[1,3]]),
        (5, [[1,0],[2,0],[3,2],[4,2],[4,3],[3,0]], []),  # No bridges
        (3, [[0,1],[1,2]], [[0,1],[1,2]]),  # Tree - all edges are bridges
    ]
    
    approaches = [
        ("Tarjan Algorithm", solution.criticalConnections_approach1_tarjan_algorithm),
        ("Naive Detection", solution.criticalConnections_approach2_naive_bridge_detection),
        ("Optimized Tarjan", solution.criticalConnections_approach3_optimized_tarjan),
        ("Iterative Tarjan", solution.criticalConnections_approach4_iterative_tarjan),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, connections, expected) in enumerate(test_cases):
            result = func(n, connections)
            
            # Normalize result for comparison (sort edges within each bridge)
            result_normalized = sorted([sorted(bridge) for bridge in result])
            expected_normalized = sorted([sorted(bridge) for bridge in expected])
            
            status = "✓" if result_normalized == expected_normalized else "✗"
            print(f"Test {i+1}: {status}")
            print(f"         Connections: {connections}")
            print(f"         Expected: {expected_normalized}")
            print(f"         Got: {result_normalized}")

def demonstrate_tarjan_algorithm():
    """Demonstrate Tarjan's bridge-finding algorithm"""
    print("\n=== Tarjan's Algorithm Demonstration ===")
    
    n = 4
    connections = [[0,1],[1,2],[2,0],[1,3]]
    
    print(f"Graph: n={n}, connections={connections}")
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    print(f"\nAdjacency list:")
    for node in range(n):
        print(f"  Node {node}: {graph[node]}")
    
    # Manual Tarjan's algorithm with tracing
    discovery = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    bridges = []
    time = [0]
    
    def tarjan_trace(u, depth=0):
        indent = "  " * depth
        print(f"{indent}Visiting node {u}")
        
        discovery[u] = low[u] = time[0]
        time[0] += 1
        
        print(f"{indent}  discovery[{u}] = low[{u}] = {discovery[u]}")
        
        for v in graph[u]:
            if discovery[v] == -1:  # Tree edge
                print(f"{indent}  Tree edge: {u} -> {v}")
                parent[v] = u
                tarjan_trace(v, depth + 1)
                
                low[u] = min(low[u], low[v])
                print(f"{indent}  Updated low[{u}] = min({low[u]}, {low[v]}) = {low[u]}")
                
                if low[v] > discovery[u]:
                    print(f"{indent}  BRIDGE FOUND: ({u}, {v}) because low[{v}]={low[v]} > discovery[{u}]={discovery[u]}")
                    bridges.append([u, v])
            
            elif v != parent[u]:  # Back edge
                print(f"{indent}  Back edge: {u} -> {v}")
                old_low = low[u]
                low[u] = min(low[u], discovery[v])
                print(f"{indent}  Updated low[{u}] = min({old_low}, {discovery[v]}) = {low[u]}")
    
    print(f"\nRunning Tarjan's algorithm:")
    for i in range(n):
        if discovery[i] == -1:
            tarjan_trace(i)
    
    print(f"\nFinal results:")
    print(f"  Discovery times: {discovery}")
    print(f"  Low-link values: {low}")
    print(f"  Bridges (critical connections): {bridges}")

def visualize_bridge_concept():
    """Visualize the concept of bridges in graphs"""
    print("\n=== Bridge Concept Visualization ===")
    
    examples = [
        ("Simple Bridge", 3, [[0,1],[1,2]], "Linear chain - all edges are bridges"),
        ("Cycle with Bridge", 4, [[0,1],[1,2],[2,0],[1,3]], "Triangle + tail - only tail edge is bridge"),
        ("No Bridges", 4, [[0,1],[1,2],[2,3],[3,0],[0,2],[1,3]], "Complete cycle - no bridges"),
        ("Multiple Bridges", 5, [[0,1],[1,2],[2,3],[3,4]], "Linear chain - all edges are bridges"),
    ]
    
    for name, n, connections, description in examples:
        print(f"\n{name}:")
        print(f"  Description: {description}")
        print(f"  Connections: {connections}")
        
        # Find bridges using Tarjan's algorithm
        solution = Solution()
        bridges = solution.criticalConnections_approach1_tarjan_algorithm(n, connections)
        
        print(f"  Bridges: {bridges}")
        print(f"  Bridge count: {len(bridges)}")
        
        # Show what happens when each bridge is removed
        for bridge in bridges:
            print(f"  Removing bridge {bridge} would disconnect the graph")

if __name__ == "__main__":
    test_critical_connections()
    demonstrate_tarjan_algorithm()
    visualize_bridge_concept()

"""
Graph Theory Concepts:
1. Bridge Detection in Undirected Graphs
2. Tarjan's Algorithm for Strongly Connected Components
3. Discovery Time and Low-Link Values
4. Critical Edge Identification

Key Tarjan's Algorithm Concepts:
- Discovery time: When node is first visited in DFS
- Low-link value: Lowest discovery time reachable from subtree
- Bridge condition: low[v] > discovery[u] for tree edge (u,v)
- Tree edges vs Back edges classification

Algorithm Insights:
- Bridge: Edge whose removal increases connected components
- Tarjan's algorithm finds all bridges in O(V+E) time
- Uses DFS with additional bookkeeping for timing
- More efficient than naive O(E*(V+E)) approach

Advanced Implementation Details:
- Handle disconnected graphs with multiple DFS starts
- Distinguish tree edges from back edges
- Avoid parent edges in undirected graphs
- Iterative implementation for large graphs

Real-world Applications:
- Network reliability analysis (critical network links)
- Transportation systems (critical roads/bridges)
- Social network analysis (key relationships)
- Circuit design (critical connections)
- Infrastructure planning (failure analysis)
- Communication networks (bottleneck identification)

Tarjan's algorithm is a fundamental tool for analyzing
structural vulnerabilities in network systems.
"""

