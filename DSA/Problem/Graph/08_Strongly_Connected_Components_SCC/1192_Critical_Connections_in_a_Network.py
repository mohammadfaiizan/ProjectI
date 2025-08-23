"""
1192. Critical Connections in a Network
Difficulty: Hard

Problem:
There are n servers numbered from 0 to n-1 connected by undirected server-to-server connections
forming a network where connections[i] = [a, b] represents a connection between servers a and b.
Any server can reach any other server directly or indirectly through the network.

A critical connection is a connection that, if removed, will make some server unable to reach some other server.

Return all critical connections in the network in any order.

Examples:
Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
Output: [[1,3]]
Explanation: [[3,1]] is also accepted.

Input: n = 2, connections = [[0,1]]
Output: [[0,1]]

Constraints:
- 1 <= n <= 10^5
- n-1 <= connections.length <= 10^5
- connections[i][0] != connections[i][1]
- There are no repeated connections.
"""

from typing import List
from collections import defaultdict

class Solution:
    def criticalConnections_approach1_tarjan_bridges(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Tarjan's Bridge-Finding Algorithm (Optimal)
        
        Use Tarjan's algorithm to find bridges (critical connections).
        A bridge is an edge whose removal increases the number of connected components.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in connections:
            graph[u].append(v)
            graph[v].append(u)
        
        # Tarjan's algorithm state
        self.time = 0
        self.discovery = [-1] * n
        self.low = [-1] * n
        self.parent = [-1] * n
        self.bridges = []
        
        def tarjan_dfs(u):
            """DFS for Tarjan's bridge finding"""
            # Initialize discovery time and low value
            self.discovery[u] = self.low[u] = self.time
            self.time += 1
            
            # Explore all neighbors
            for v in graph[u]:
                if self.discovery[v] == -1:  # Tree edge
                    self.parent[v] = u
                    tarjan_dfs(v)
                    
                    # Update low value
                    self.low[u] = min(self.low[u], self.low[v])
                    
                    # Check if edge (u,v) is a bridge
                    if self.low[v] > self.discovery[u]:
                        self.bridges.append([u, v])
                        
                elif v != self.parent[u]:  # Back edge (not to parent)
                    self.low[u] = min(self.low[u], self.discovery[v])
        
        # Run DFS from all unvisited vertices
        for i in range(n):
            if self.discovery[i] == -1:
                tarjan_dfs(i)
        
        return self.bridges
    
    def criticalConnections_approach2_naive_removal(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Naive Edge Removal (Brute Force)
        
        Try removing each edge and check if graph becomes disconnected.
        
        Time: O(E * (V + E)) - Too slow for large inputs
        Space: O(V + E)
        """
        def is_connected(excluded_edge):
            """Check if graph is connected after removing an edge"""
            # Build graph excluding the specified edge
            temp_graph = defaultdict(list)
            for u, v in connections:
                if [u, v] != excluded_edge and [v, u] != excluded_edge:
                    temp_graph[u].append(v)
                    temp_graph[v].append(u)
            
            # DFS to check connectivity
            visited = set()
            stack = [0]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                
                for neighbor in temp_graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            return len(visited) == n
        
        critical = []
        for edge in connections:
            if not is_connected(edge):
                critical.append(edge)
        
        return critical
    
    def criticalConnections_approach3_union_find_bridges(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Union-Find with Bridge Detection
        
        Use Union-Find to detect bridges by checking connectivity after edge removal.
        
        Time: O(E * α(V)) - Still slower than Tarjan's
        Space: O(V)
        """
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.components = n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                
                if self.rank[px] < self.rank[py]:
                    self.parent[px] = py
                elif self.rank[px] > self.rank[py]:
                    self.parent[py] = px
                else:
                    self.parent[py] = px
                    self.rank[px] += 1
                
                self.components -= 1
                return True
            
            def is_connected(self):
                return self.components == 1
        
        def is_bridge(excluded_edge):
            """Check if removing edge makes graph disconnected"""
            uf = UnionFind(n)
            
            for u, v in connections:
                if [u, v] != excluded_edge and [v, u] != excluded_edge:
                    uf.union(u, v)
            
            return not uf.is_connected()
        
        bridges = []
        for edge in connections:
            if is_bridge(edge):
                bridges.append(edge)
        
        return bridges
    
    def criticalConnections_approach4_dfs_articulation_edges(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: DFS-based Articulation Edge Finding
        
        Modified DFS to find articulation edges (bridges).
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in connections:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = [False] * n
        discovery = [0] * n
        low = [0] * n
        parent = [-1] * n
        bridges = []
        self.time = 0
        
        def bridge_dfs(u):
            """DFS to find bridges"""
            visited[u] = True
            discovery[u] = low[u] = self.time
            self.time += 1
            
            for v in graph[u]:
                if not visited[v]:
                    parent[v] = u
                    bridge_dfs(v)
                    
                    # Update low value
                    low[u] = min(low[u], low[v])
                    
                    # Check if edge u-v is a bridge
                    if low[v] > discovery[u]:
                        bridges.append([u, v])
                        
                elif v != parent[u]:  # Back edge
                    low[u] = min(low[u], discovery[v])
        
        # Run DFS from all unvisited vertices
        for i in range(n):
            if not visited[i]:
                bridge_dfs(i)
        
        return bridges
    
    def criticalConnections_approach5_iterative_tarjan(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        """
        Approach 5: Iterative Tarjan's Algorithm
        
        Iterative implementation to avoid recursion stack overflow.
        
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
        time = [0]  # Use list to allow modification in nested function
        
        def iterative_tarjan_dfs(start):
            """Iterative DFS for bridge finding"""
            # Stack: (node, neighbor_index, is_returning)
            stack = [(start, 0, False)]
            
            while stack:
                u, neighbor_idx, is_returning = stack.pop()
                
                if is_returning:
                    # Returning from processing a neighbor
                    if neighbor_idx > 0:
                        neighbors = list(graph[u])
                        if neighbor_idx <= len(neighbors):
                            v = neighbors[neighbor_idx - 1]
                            if parent[v] == u:  # Tree edge
                                low[u] = min(low[u], low[v])
                                
                                # Check if bridge
                                if low[v] > discovery[u]:
                                    bridges.append([u, v])
                    
                    # Continue with next neighbor
                    neighbors = list(graph[u])
                    while neighbor_idx < len(neighbors):
                        v = neighbors[neighbor_idx]
                        neighbor_idx += 1
                        
                        if discovery[v] == -1:  # Tree edge
                            parent[v] = u
                            stack.append((u, neighbor_idx, True))
                            stack.append((v, 0, False))
                            break
                        elif v != parent[u]:  # Back edge
                            low[u] = min(low[u], discovery[v])
                else:
                    # First time visiting node
                    if discovery[u] != -1:
                        continue
                    
                    discovery[u] = low[u] = time[0]
                    time[0] += 1
                    
                    # Process neighbors
                    neighbors = list(graph[u])
                    if neighbors:
                        v = neighbors[0]
                        if discovery[v] == -1:
                            parent[v] = u
                            stack.append((u, 1, True))
                            stack.append((v, 0, False))
                        else:
                            if v != parent[u]:
                                low[u] = min(low[u], discovery[v])
                            stack.append((u, 1, True))
        
        # Run DFS from all unvisited vertices
        for i in range(n):
            if discovery[i] == -1:
                iterative_tarjan_dfs(i)
        
        return bridges

def test_critical_connections():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (n, connections, expected_bridges)
        (4, [[0,1],[1,2],[2,0],[1,3]], [[1,3]]),
        (2, [[0,1]], [[0,1]]),
        (6, [[0,1],[1,2],[2,0],[1,3],[3,4],[4,5],[5,3]], [[1,3]]),
        (5, [[1,0],[2,0],[3,2],[4,2]], [[0,1],[0,2]]),
        (3, [[0,1],[1,2]], [[0,1],[1,2]]),
    ]
    
    approaches = [
        ("Tarjan's Bridges", solution.criticalConnections_approach1_tarjan_bridges),
        ("DFS Articulation", solution.criticalConnections_approach4_dfs_articulation_edges),
        ("Iterative Tarjan", solution.criticalConnections_approach5_iterative_tarjan),
        # Skip naive approaches for performance
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, connections, expected) in enumerate(test_cases):
            result = func(n, connections[:])  # Deep copy
            
            # Sort for comparison
            result_sorted = sorted([sorted(edge) for edge in result])
            expected_sorted = sorted([sorted(edge) for edge in expected])
            
            status = "✓" if result_sorted == expected_sorted else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={len(expected)}, got={len(result)}")
            if result_sorted != expected_sorted:
                print(f"  Expected: {expected_sorted}")
                print(f"  Got: {result_sorted}")

def demonstrate_bridge_finding():
    """Demonstrate bridge finding algorithm step by step"""
    print("\n=== Bridge Finding Demo ===")
    
    n = 5
    connections = [[0,1],[1,2],[2,0],[1,3],[3,4]]
    
    print(f"Graph: n={n}, connections={connections}")
    
    # Visualize graph structure
    print(f"\nGraph structure:")
    print(f"  Triangle: 0-1-2-0")
    print(f"  Bridge: 1-3 (connects triangle to linear part)")
    print(f"  Edge: 3-4")
    
    # Build adjacency list for visualization
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    print(f"\nAdjacency list:")
    for vertex in sorted(graph.keys()):
        print(f"  {vertex}: {sorted(graph[vertex])}")
    
    # Find bridges
    solution = Solution()
    bridges = solution.criticalConnections_approach1_tarjan_bridges(n, connections)
    
    print(f"\nCritical connections (bridges): {bridges}")
    
    # Explain why each is a bridge
    for bridge in bridges:
        u, v = bridge
        print(f"\nBridge {u}-{v}:")
        print(f"  Removing this edge would disconnect vertex {v} from the rest")
        print(f"  This is the only path between the triangle {0,1,2} and vertices {3,4}")

def demonstrate_tarjan_algorithm():
    """Demonstrate Tarjan's algorithm execution"""
    print("\n=== Tarjan's Algorithm Demo ===")
    
    connections = [[0,1],[1,2],[2,0],[1,3]]
    n = 4
    
    print(f"Graph: {connections}")
    
    # Simulate algorithm execution
    print(f"\nTarjan's Algorithm Execution:")
    print(f"1. Start DFS from vertex 0")
    print(f"   discovery[0] = 0, low[0] = 0")
    
    print(f"2. Visit neighbor 1")
    print(f"   discovery[1] = 1, low[1] = 1")
    
    print(f"3. From 1, visit neighbor 2")
    print(f"   discovery[2] = 2, low[2] = 2")
    
    print(f"4. From 2, back edge to 0")
    print(f"   low[2] = min(low[2], discovery[0]) = min(2, 0) = 0")
    
    print(f"5. Return to 1, update low[1]")
    print(f"   low[1] = min(low[1], low[2]) = min(1, 0) = 0")
    
    print(f"6. From 1, visit neighbor 3")
    print(f"   discovery[3] = 3, low[3] = 3")
    
    print(f"7. No neighbors from 3, return to 1")
    print(f"   low[1] remains 0, but low[3] = 3 > discovery[1] = 1")
    print(f"   Therefore, edge (1,3) is a bridge!")
    
    # Verify with actual algorithm
    solution = Solution()
    bridges = solution.criticalConnections_approach1_tarjan_bridges(n, connections)
    print(f"\nActual result: {bridges}")

def analyze_bridge_properties():
    """Analyze properties of bridges in graphs"""
    print("\n=== Bridge Properties Analysis ===")
    
    print("Bridge Characteristics:")
    
    print("\n1. **Definition:**")
    print("   • Edge whose removal increases connected components")
    print("   • Critical for maintaining graph connectivity")
    print("   • Also called 'articulation edges'")
    
    print("\n2. **Detection Condition:**")
    print("   • For edge (u,v) in DFS tree:")
    print("   • Bridge if low[v] > discovery[u]")
    print("   • Means no back edge from subtree of v to ancestors of u")
    
    print("\n3. **Graph Types:**")
    print("   • Trees: All edges are bridges")
    print("   • Cycles: No bridges")
    print("   • Bridge-connected components: Connected by bridges")
    
    print("\n4. **Applications:**")
    print("   • Network reliability analysis")
    print("   • Critical infrastructure identification")
    print("   • Circuit analysis and fault tolerance")
    print("   • Social network analysis")
    
    print("\n5. **Related Concepts:**")
    print("   • Articulation points (vertices)")
    print("   • 2-edge-connected components")
    print("   • Block-cut tree structure")
    print("   • Network flow bottlenecks")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of critical connections"""
    print("\n=== Real-World Applications ===")
    
    print("Critical Connection Applications:")
    
    print("\n1. **Network Infrastructure:**")
    print("   • Internet backbone reliability")
    print("   • Telecommunications network design")
    print("   • Power grid critical transmission lines")
    print("   • Transportation network bottlenecks")
    
    print("\n2. **Software Systems:**")
    print("   • Distributed system fault tolerance")
    print("   • Microservice dependency analysis")
    print("   • Database replication topology")
    print("   • Content delivery network optimization")
    
    print("\n3. **Social Networks:**")
    print("   • Influential relationship identification")
    print("   • Information flow bottlenecks")
    print("   • Community structure analysis")
    print("   • Viral spread prevention")
    
    print("\n4. **Transportation:**")
    print("   • Critical road/bridge identification")
    print("   • Public transit reliability")
    print("   • Supply chain vulnerability")
    print("   • Emergency evacuation planning")
    
    print("\n5. **Biological Networks:**")
    print("   • Protein interaction networks")
    print("   • Metabolic pathway analysis")
    print("   • Food web critical species")
    print("   • Disease transmission networks")
    
    print("\nKey Benefits:")
    print("• Identify single points of failure")
    print("• Prioritize infrastructure investment")
    print("• Design fault-tolerant systems")
    print("• Optimize network reliability")

def compare_bridge_algorithms():
    """Compare different bridge-finding algorithms"""
    print("\n=== Bridge Algorithm Comparison ===")
    
    print("Algorithm Performance Comparison:")
    
    print("\n1. **Tarjan's Bridge Algorithm:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V)")
    print("   • Single DFS pass")
    print("   • Industry standard")
    print("   • Optimal complexity")
    
    print("\n2. **Naive Edge Removal:**")
    print("   • Time: O(E × (V + E))")
    print("   • Space: O(V + E)")
    print("   • Simple but inefficient")
    print("   • Good for understanding")
    print("   • Not practical for large graphs")
    
    print("\n3. **Union-Find Approach:**")
    print("   • Time: O(E × α(V))")
    print("   • Space: O(V)")
    print("   • Better than naive")
    print("   • Still slower than Tarjan's")
    print("   • Useful for dynamic scenarios")
    
    print("\n4. **Iterative Implementations:**")
    print("   • Same complexity as recursive")
    print("   • Avoid stack overflow")
    print("   • More complex code")
    print("   • Better for production")
    
    print("\nSelection Guidelines:")
    print("• **Tarjan's:** General purpose, optimal performance")
    print("• **Iterative:** Deep graphs, stack overflow concerns")
    print("• **Naive:** Educational, very small graphs")
    print("• **Union-Find:** Dynamic graph scenarios")

if __name__ == "__main__":
    test_critical_connections()
    demonstrate_bridge_finding()
    demonstrate_tarjan_algorithm()
    analyze_bridge_properties()
    demonstrate_real_world_applications()
    compare_bridge_algorithms()

"""
Critical Connections and Bridge Finding Concepts:
1. Tarjan's Bridge-Finding Algorithm with Low-Link Values
2. DFS Tree and Back Edge Analysis for Bridge Detection
3. Network Reliability and Single Point of Failure Analysis
4. Graph Connectivity and Articulation Edge Identification
5. Real-time Bridge Detection in Dynamic Networks

Key Problem Insights:
- Bridges are edges whose removal disconnects the graph
- Tarjan's algorithm finds all bridges in O(V + E) time
- Bridge detection essential for network reliability analysis
- Critical for identifying infrastructure vulnerabilities

Algorithm Strategy:
1. DFS traversal with discovery and low-link time tracking
2. Identify tree edges vs back edges during traversal
3. Bridge condition: low[v] > discovery[u] for edge (u,v)
4. Back edges indicate alternative paths (no bridge)

Bridge Detection:
- Tree edge (u,v) is bridge if no back edge from subtree of v
- Low-link value tracks earliest reachable ancestor
- Discovery time provides DFS ordering reference
- Efficient single-pass algorithm with optimal complexity

Network Analysis Applications:
- Internet backbone and telecommunications reliability
- Power grid critical transmission line identification
- Transportation network bottleneck analysis
- Social network influence and information flow
- Distributed system fault tolerance design

Real-world Impact:
- Infrastructure investment prioritization
- Fault-tolerant system design
- Emergency planning and response
- Network optimization and redundancy
- Security and vulnerability assessment

This implementation provides complete bridge detection mastery
essential for network reliability and infrastructure analysis.
"""
