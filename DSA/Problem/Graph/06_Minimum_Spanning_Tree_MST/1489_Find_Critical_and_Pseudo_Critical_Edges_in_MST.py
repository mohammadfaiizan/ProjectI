"""
1489. Find Critical and Pseudo-Critical Edges in MST
Difficulty: Hard

Problem:
Given a weighted undirected connected graph with n vertices numbered from 0 to n - 1, 
and an array edges where edges[i] = [ai, bi, weighti] represents a bidirectional and 
weighted edge between vertices ai and bi. A minimum spanning tree (MST) is a subset 
of the graph's edges that connects all vertices without cycles and with the minimum 
possible total edge weight.

Find all the critical and pseudo-critical edges in the given graph's MST. An MST edge 
whose deletion from the graph would cause the MST weight to increase is called a 
critical edge. A pseudo-critical edge, on the other hand, is that which can appear 
in some MSTs but not all.

Note that you can return the indices of the edges in any order.

Examples:
Input: n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
Output: [[0,1],[2,3,4,5]]

Input: n = 4, edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,1]]
Output: [[],[0,1,2,3]]

Constraints:
- 2 <= n <= 100
- 1 <= edges.length <= min(200, n * (n - 1) / 2)
- edges[i].length == 3
- 0 <= ai < bi < n
- 1 <= weighti <= 1000
- All pairs (ai, bi) are distinct
"""

from typing import List

class Solution:
    def findCriticalAndPseudoCriticalEdges_approach1_mst_validation(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: MST Validation (Optimal)
        
        For each edge, check if it's critical by removing it and pseudo-critical by forcing it.
        
        Time: O(E^2 * α(V)) where α is inverse Ackermann function
        Space: O(V)
        """
        # Add original indices to edges
        indexed_edges = [(edges[i][2], edges[i][0], edges[i][1], i) for i in range(len(edges))]
        
        # Find MST weight using Kruskal's algorithm
        mst_weight = self._kruskal_mst_weight(n, indexed_edges)
        
        critical = []
        pseudo_critical = []
        
        for i in range(len(edges)):
            # Check if edge is critical
            # Remove edge i and see if MST weight increases
            edges_without_i = [indexed_edges[j] for j in range(len(indexed_edges)) if j != i]
            weight_without_i = self._kruskal_mst_weight(n, edges_without_i)
            
            if weight_without_i > mst_weight:
                critical.append(i)
            else:
                # Check if edge is pseudo-critical
                # Force edge i to be in MST and see if weight remains same
                weight_with_i = self._kruskal_mst_weight_with_forced_edge(n, indexed_edges, i)
                
                if weight_with_i == mst_weight:
                    pseudo_critical.append(i)
        
        return [critical, pseudo_critical]
    
    def _kruskal_mst_weight(self, n: int, indexed_edges: List[tuple]) -> int:
        """Calculate MST weight using Kruskal's algorithm"""
        # Union-Find
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Sort edges by weight
        sorted_edges = sorted(indexed_edges)
        
        weight = 0
        edges_used = 0
        
        for edge_weight, u, v, _ in sorted_edges:
            if union(u, v):
                weight += edge_weight
                edges_used += 1
                
                if edges_used == n - 1:
                    break
        
        # Return infinity if graph is not connected
        return weight if edges_used == n - 1 else float('inf')
    
    def _kruskal_mst_weight_with_forced_edge(self, n: int, indexed_edges: List[tuple], forced_edge_idx: int) -> int:
        """Calculate MST weight with a forced edge"""
        # Union-Find
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Force the edge at forced_edge_idx
        forced_edge = indexed_edges[forced_edge_idx]
        weight, u, v, _ = forced_edge
        
        if not union(u, v):
            # Forced edge creates cycle with itself, invalid
            return float('inf')
        
        total_weight = weight
        edges_used = 1
        
        # Sort remaining edges by weight
        remaining_edges = [indexed_edges[i] for i in range(len(indexed_edges)) if i != forced_edge_idx]
        remaining_edges.sort()
        
        for edge_weight, u, v, _ in remaining_edges:
            if union(u, v):
                total_weight += edge_weight
                edges_used += 1
                
                if edges_used == n - 1:
                    break
        
        return total_weight if edges_used == n - 1 else float('inf')
    
    def findCriticalAndPseudoCriticalEdges_approach2_optimized_validation(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Optimized Validation with Early Termination
        
        Enhanced version with optimizations for better performance.
        
        Time: O(E^2 * α(V))
        Space: O(V + E)
        """
        # Add indices and sort by weight
        indexed_edges = []
        for i, (u, v, w) in enumerate(edges):
            indexed_edges.append((w, u, v, i))
        
        indexed_edges.sort()
        
        # Find MST weight
        mst_weight = self._find_mst_weight(n, indexed_edges)
        
        critical = []
        pseudo_critical = []
        
        for i in range(len(edges)):
            original_idx = indexed_edges[i][3] if i < len(indexed_edges) else i
            
            # Find original index in sorted array
            edge_idx_in_sorted = -1
            for j, (_, _, _, idx) in enumerate(indexed_edges):
                if idx == i:
                    edge_idx_in_sorted = j
                    break
            
            if edge_idx_in_sorted == -1:
                continue
            
            # Test if critical (remove edge)
            edges_without = indexed_edges[:edge_idx_in_sorted] + indexed_edges[edge_idx_in_sorted + 1:]
            weight_without = self._find_mst_weight(n, edges_without)
            
            if weight_without > mst_weight:
                critical.append(i)
            else:
                # Test if pseudo-critical (force edge)
                weight_with_forced = self._find_mst_weight_forced(n, indexed_edges, edge_idx_in_sorted)
                
                if weight_with_forced == mst_weight:
                    pseudo_critical.append(i)
        
        return [critical, pseudo_critical]
    
    def _find_mst_weight(self, n: int, edges: List[tuple]) -> int:
        """Find MST weight using Kruskal's algorithm"""
        uf = UnionFind(n)
        weight = 0
        
        for w, u, v, _ in edges:
            if uf.union(u, v):
                weight += w
                if uf.components == 1:
                    break
        
        return weight if uf.components == 1 else float('inf')
    
    def _find_mst_weight_forced(self, n: int, edges: List[tuple], forced_idx: int) -> int:
        """Find MST weight with forced edge"""
        uf = UnionFind(n)
        
        # Force the edge
        w, u, v, _ = edges[forced_idx]
        if not uf.union(u, v):
            return float('inf')  # Creates cycle
        
        weight = w
        
        # Add other edges
        for i, (w, u, v, _) in enumerate(edges):
            if i != forced_idx and uf.union(u, v):
                weight += w
                if uf.components == 1:
                    break
        
        return weight if uf.components == 1 else float('inf')
    
    def findCriticalAndPseudoCriticalEdges_approach3_tarjan_bridge_finding(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Tarjan's Bridge Finding Algorithm (Advanced)
        
        Use bridge finding to identify critical edges in MST.
        
        Time: O(V + E) for bridge finding + O(E^2) for validation
        Space: O(V + E)
        """
        # First find MST using standard algorithm
        indexed_edges = [(edges[i][2], edges[i][0], edges[i][1], i) for i in range(len(edges))]
        mst_weight, mst_edges = self._find_mst_edges(n, indexed_edges)
        
        if mst_weight == float('inf'):
            return [[], []]
        
        # Build MST graph
        mst_graph = [[] for _ in range(n)]
        mst_edge_indices = set()
        
        for u, v, original_idx in mst_edges:
            mst_graph[u].append(v)
            mst_graph[v].append(u)
            mst_edge_indices.add(original_idx)
        
        # Find bridges in MST using Tarjan's algorithm
        bridges = self._find_bridges(n, mst_graph, mst_edges)
        
        critical = []
        pseudo_critical = []
        
        for i in range(len(edges)):
            if i in mst_edge_indices:
                # Edge is in MST
                if self._is_bridge_edge(edges[i][0], edges[i][1], bridges):
                    critical.append(i)
                else:
                    pseudo_critical.append(i)
            else:
                # Edge not in MST, check if it can be in some MST
                weight_with_forced = self._find_mst_weight_forced_by_index(n, indexed_edges, i)
                if weight_with_forced == mst_weight:
                    pseudo_critical.append(i)
        
        return [critical, pseudo_critical]
    
    def _find_mst_edges(self, n: int, indexed_edges: List[tuple]) -> tuple:
        """Find MST edges and weight"""
        uf = UnionFind(n)
        sorted_edges = sorted(indexed_edges)
        
        weight = 0
        mst_edges = []
        
        for w, u, v, idx in sorted_edges:
            if uf.union(u, v):
                weight += w
                mst_edges.append((u, v, idx))
                if len(mst_edges) == n - 1:
                    break
        
        return weight if len(mst_edges) == n - 1 else float('inf'), mst_edges
    
    def _find_bridges(self, n: int, graph: List[List[int]], edges: List[tuple]) -> List[tuple]:
        """Find bridges using Tarjan's algorithm"""
        visited = [False] * n
        disc = [0] * n
        low = [0] * n
        parent = [-1] * n
        bridges = []
        time = [0]
        
        def bridge_dfs(u):
            visited[u] = True
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in graph[u]:
                if not visited[v]:
                    parent[v] = u
                    bridge_dfs(v)
                    low[u] = min(low[u], low[v])
                    
                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        
        for i in range(n):
            if not visited[i]:
                bridge_dfs(i)
        
        return bridges
    
    def _is_bridge_edge(self, u: int, v: int, bridges: List[tuple]) -> bool:
        """Check if edge is a bridge"""
        return (u, v) in bridges or (v, u) in bridges
    
    def _find_mst_weight_forced_by_index(self, n: int, indexed_edges: List[tuple], force_original_idx: int) -> int:
        """Find MST weight with forced edge by original index"""
        # Find the edge with original index
        forced_edge = None
        for w, u, v, idx in indexed_edges:
            if idx == force_original_idx:
                forced_edge = (w, u, v, idx)
                break
        
        if not forced_edge:
            return float('inf')
        
        uf = UnionFind(n)
        w, u, v, _ = forced_edge
        
        if not uf.union(u, v):
            return float('inf')
        
        weight = w
        remaining_edges = [e for e in indexed_edges if e[3] != force_original_idx]
        remaining_edges.sort()
        
        for w, u, v, _ in remaining_edges:
            if uf.union(u, v):
                weight += w
                if uf.components == 1:
                    break
        
        return weight if uf.components == 1 else float('inf')

class UnionFind:
    """Optimized Union-Find with path compression and union by rank"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.components -= 1
        return True

def test_critical_pseudo_critical_edges():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, expected)
        (5, [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]], [[0,1],[2,3,4,5]]),
        (4, [[0,1,1],[1,2,1],[2,3,1],[0,3,1]], [[],[0,1,2,3]]),
        (4, [[0,1,1],[1,2,1],[2,3,1]], [[0,1,2],[]]),
        (3, [[0,1,1],[1,2,2],[0,2,2]], [[0],[1,2]]),
        (6, [[0,1,1],[1,2,1],[0,2,1],[2,3,4],[3,4,2],[3,5,2],[4,5,2]], [[3],[0,1,2,4,5,6]]),
    ]
    
    approaches = [
        ("MST Validation", solution.findCriticalAndPseudoCriticalEdges_approach1_mst_validation),
        ("Optimized Validation", solution.findCriticalAndPseudoCriticalEdges_approach2_optimized_validation),
        ("Tarjan Bridge Finding", solution.findCriticalAndPseudoCriticalEdges_approach3_tarjan_bridge_finding),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, expected) in enumerate(test_cases):
            result = func(n, edges[:])  # Deep copy
            
            # Sort for comparison
            result[0].sort()
            result[1].sort()
            expected[0].sort()
            expected[1].sort()
            
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Expected: {expected}, Got: {result}")

def demonstrate_critical_edge_concept():
    """Demonstrate critical and pseudo-critical edge concepts"""
    print("\n=== Critical and Pseudo-Critical Edges Demo ===")
    
    n = 4
    edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,1]]
    
    print(f"Graph: {n} vertices, edges: {edges}")
    print(f"Goal: Identify critical and pseudo-critical edges")
    
    # Find MST weight
    indexed_edges = [(edges[i][2], edges[i][0], edges[i][1], i) for i in range(len(edges))]
    indexed_edges.sort()
    
    print(f"\nSorted edges by weight: {indexed_edges}")
    
    # Demonstrate MST construction
    print(f"\nMST Construction:")
    uf = UnionFind(n)
    mst_weight = 0
    mst_edges = []
    
    for w, u, v, idx in indexed_edges:
        if uf.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, idx))
            print(f"  Add edge {idx}: ({u},{v}) weight {w}")
        else:
            print(f"  Skip edge {idx}: ({u},{v}) weight {w} (creates cycle)")
    
    print(f"MST weight: {mst_weight}")
    print(f"MST edges: {[(u,v) for u,v,_ in mst_edges]}")
    
    # Analyze each edge
    print(f"\nEdge Analysis:")
    for i in range(len(edges)):
        u, v, w = edges[i]
        print(f"\nEdge {i}: ({u},{v}) weight {w}")
        
        # Test if critical (remove edge)
        edges_without = [indexed_edges[j] for j in range(len(indexed_edges)) 
                        if indexed_edges[j][3] != i]
        
        uf_test = UnionFind(n)
        weight_without = 0
        edges_added = 0
        
        for w_test, u_test, v_test, _ in edges_without:
            if uf_test.union(u_test, v_test):
                weight_without += w_test
                edges_added += 1
        
        is_connected_without = (edges_added == n - 1)
        
        if not is_connected_without or weight_without > mst_weight:
            print(f"  CRITICAL: Removing disconnects graph or increases MST weight")
        else:
            # Test if pseudo-critical (force edge)
            uf_forced = UnionFind(n)
            if uf_forced.union(u, v):
                weight_forced = w
                
                for w_test, u_test, v_test, idx_test in indexed_edges:
                    if idx_test != i and uf_forced.union(u_test, v_test):
                        weight_forced += w_test
                
                if weight_forced == mst_weight:
                    print(f"  PSEUDO-CRITICAL: Can be in some MST")
                else:
                    print(f"  NEITHER: Not useful for MST")
            else:
                print(f"  NEITHER: Creates cycle when forced")

def analyze_mst_edge_classification():
    """Analyze MST edge classification theory"""
    print("\n=== MST Edge Classification Theory ===")
    
    print("Edge Classification in MST:")
    
    print("\n1. **Critical Edges:**")
    print("   • Removing the edge increases MST weight")
    print("   • Must be in every MST of the graph")
    print("   • Bridge edges in MST are always critical")
    print("   • Essential for maintaining connectivity")
    
    print("\n2. **Pseudo-Critical Edges:**")
    print("   • Can appear in some MSTs but not all")
    print("   • Removing doesn't increase MST weight")
    print("   • Forcing inclusion doesn't increase MST weight")
    print("   • Alternative edges with same optimization potential")
    
    print("\n3. **Non-Critical Edges:**")
    print("   • Cannot appear in any MST")
    print("   • Either creates cycles or increases weight")
    print("   • Redundant for MST construction")
    
    print("\nIdentification Algorithm:")
    print("For each edge e:")
    print("1. Remove e and compute MST weight")
    print("   • If weight increases → CRITICAL")
    print("   • If weight same → continue to step 2")
    print("2. Force e and compute MST weight")
    print("   • If weight same as original → PSEUDO-CRITICAL")
    print("   • If weight increases → NON-CRITICAL")
    
    print("\nTheoretical Foundation:")
    print("• Based on cut property and cycle property of MST")
    print("• Critical edges are bridges in MST")
    print("• Pseudo-critical edges offer alternative optimal paths")
    print("• Classification helps in network robustness analysis")

def demonstrate_bridge_finding():
    """Demonstrate bridge finding in MST"""
    print("\n=== Bridge Finding in MST Demo ===")
    
    print("Tarjan's Bridge Finding Algorithm:")
    print("• DFS-based algorithm to find bridges")
    print("• Bridge: edge whose removal increases connected components")
    print("• In MST context: bridges are critical edges")
    
    print("\nAlgorithm Steps:")
    print("1. DFS traversal with discovery times")
    print("2. Track low-link values (earliest reachable vertex)")
    print("3. Edge (u,v) is bridge if low[v] > disc[u]")
    print("4. All bridges in MST are critical edges")
    
    print("\nTime Complexity:")
    print("• Tarjan's algorithm: O(V + E)")
    print("• MST construction: O(E log E)")
    print("• Total: O(E log E) for complete analysis")
    
    print("\nAdvantages:")
    print("• Efficient for identifying critical edges")
    print("• Linear time bridge detection")
    print("• Natural integration with MST algorithms")

def analyze_network_robustness():
    """Analyze network robustness using edge classification"""
    print("\n=== Network Robustness Analysis ===")
    
    print("Applications in Network Design:")
    
    print("\n1. **Infrastructure Planning:**")
    print("   • Critical edges: Must have redundant backup")
    print("   • Pseudo-critical: Alternative routing options")
    print("   • Non-critical: Can be omitted to save cost")
    
    print("\n2. **Fault Tolerance:**")
    print("   • Critical edge failure: Major network disruption")
    print("   • Pseudo-critical failure: Alternative paths available")
    print("   • Design redundancy for critical connections")
    
    print("\n3. **Cost Optimization:**")
    print("   • Focus resources on critical edges")
    print("   • Pseudo-critical edges provide flexibility")
    print("   • Remove non-critical edges to reduce cost")
    
    print("\n4. **Maintenance Scheduling:**")
    print("   • Critical edges: Require immediate attention")
    print("   • Pseudo-critical: Scheduled maintenance possible")
    print("   • Priority-based maintenance resource allocation")
    
    print("\nRobustness Metrics:")
    print("• Critical edge ratio: |Critical| / |MST edges|")
    print("• Redundancy factor: |Pseudo-critical| / |Total edges|")
    print("• Connectivity resilience: Alternative path availability")
    print("• Recovery time: Time to restore after critical edge failure")
    
    print("\nDesign Guidelines:")
    print("• Minimize critical edges through strategic redundancy")
    print("• Maximize pseudo-critical options for flexibility")
    print("• Balance cost, robustness, and performance")
    print("• Plan for graceful degradation under failures")

if __name__ == "__main__":
    test_critical_pseudo_critical_edges()
    demonstrate_critical_edge_concept()
    analyze_mst_edge_classification()
    demonstrate_bridge_finding()
    analyze_network_robustness()

"""
MST Critical and Pseudo-Critical Edges Concepts:
1. Edge Classification in Minimum Spanning Trees
2. MST Validation and Testing Algorithms
3. Tarjan's Bridge Finding for Critical Edge Detection
4. Network Robustness and Fault Tolerance Analysis
5. Union-Find Optimization for Efficient MST Operations

Key Problem Insights:
- Critical edges: must be in every MST, removal increases weight
- Pseudo-critical edges: can be in some MSTs, provide alternatives
- Non-critical edges: cannot be in any MST optimally
- Classification enables network robustness analysis

Algorithm Strategy:
1. Find original MST weight using Kruskal's algorithm
2. For each edge: test removal (critical) and forcing (pseudo-critical)
3. Use optimized Union-Find for efficient connectivity testing
4. Alternative: Tarjan's bridge finding for critical edges

Edge Testing Methodology:
- Remove edge and check if MST weight increases (critical test)
- Force edge inclusion and check if weight remains same (pseudo-critical test)
- Use Union-Find for efficient cycle detection and connectivity
- Early termination optimizations for large graphs

MST Theory Applications:
- Cut property: minimum edge crossing cut is in MST
- Cycle property: maximum edge in cycle not in MST
- Bridge edges in MST are always critical
- Edge weight equality creates multiple valid MSTs

Union-Find Optimizations:
- Path compression for find operations
- Union by rank for balanced tree structure
- Component counting for connectivity verification
- Amortized O(α(n)) operations where α is inverse Ackermann

Real-world Applications:
- Network infrastructure robustness analysis
- Critical path identification in supply chains
- Fault tolerance planning in distributed systems
- Redundancy optimization in communication networks
- Maintenance prioritization in transportation systems

This problem demonstrates advanced MST analysis
for network robustness and reliability assessment.
"""
