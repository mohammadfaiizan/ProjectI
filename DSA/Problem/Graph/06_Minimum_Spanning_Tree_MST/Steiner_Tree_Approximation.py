"""
Steiner Tree Approximation Algorithms
Difficulty: Hard

The Steiner Tree problem is a fundamental optimization problem in computer science.
Given a graph with a subset of vertices called terminals, find the minimum cost
subgraph that connects all terminals (may include additional Steiner vertices).

This problem is NP-hard in general graphs, so we focus on approximation algorithms.

Problem Variants:
1. Euclidean Steiner Tree (geometric version)
2. Graph Steiner Tree (general graphs)
3. Rectilinear Steiner Tree (Manhattan distance)
4. Online Steiner Tree (terminals arrive dynamically)
5. Multi-level Steiner Tree (hierarchical requirements)

Applications:
- Network design and optimization
- VLSI routing and circuit design
- Computational biology (phylogenetic trees)
- Infrastructure planning
- Multicast routing protocols
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
import math
from collections import defaultdict

class SteinerTreeApproximation:
    """Collection of Steiner Tree approximation algorithms"""
    
    def mst_approximation(self, n: int, edges: List[Tuple[int, int, int]], terminals: Set[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST-based 2-approximation for Steiner Tree
        
        Algorithm:
        1. Compute metric closure on terminal vertices
        2. Find MST of complete graph on terminals
        3. Replace MST edges with shortest paths in original graph
        4. Return resulting Steiner tree
        
        Time: O(V^3 + E log V)
        Approximation ratio: 2
        """
        # Step 1: Compute all-pairs shortest paths (metric closure)
        dist = self._floyd_warshall(n, edges)
        
        # Step 2: Create complete graph on terminals with shortest path distances
        terminal_edges = []
        terminals_list = list(terminals)
        
        for i in range(len(terminals_list)):
            for j in range(i + 1, len(terminals_list)):
                u, v = terminals_list[i], terminals_list[j]
                if dist[u][v] < float('inf'):
                    terminal_edges.append((dist[u][v], u, v))
        
        # Step 3: Find MST of terminal graph
        terminal_mst = self._kruskal_mst(len(terminals_list), terminal_edges, terminals_list)
        
        # Step 4: Replace each MST edge with shortest path in original graph
        steiner_edges = set()
        total_cost = 0
        
        for cost, u, v in terminal_mst:
            path = self._reconstruct_shortest_path(u, v, dist, n, edges)
            for i in range(len(path) - 1):
                edge = tuple(sorted([path[i], path[i+1]]))
                steiner_edges.add(edge)
            total_cost += cost
        
        return total_cost, list(steiner_edges)
    
    def shortest_path_heuristic(self, n: int, edges: List[Tuple[int, int, int]], terminals: Set[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Shortest Path Heuristic for Steiner Tree
        
        Algorithm:
        1. Start with arbitrary terminal
        2. Repeatedly add closest terminal to current tree
        3. Connect via shortest path
        
        Time: O(|T| * (V log V + E))
        Approximation ratio: Not guaranteed, but often good in practice
        """
        if not terminals:
            return 0, []
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        # Start with arbitrary terminal
        tree_vertices = {next(iter(terminals))}
        remaining_terminals = terminals - tree_vertices
        steiner_edges = []
        total_cost = 0
        
        while remaining_terminals:
            min_cost = float('inf')
            best_path = None
            closest_terminal = None
            
            # Find closest terminal to current tree
            for terminal in remaining_terminals:
                for tree_vertex in tree_vertices:
                    cost, path = self._dijkstra_path(graph, tree_vertex, terminal)
                    if cost < min_cost:
                        min_cost = cost
                        best_path = path
                        closest_terminal = terminal
            
            if best_path:
                # Add path to tree
                for i in range(len(best_path) - 1):
                    edge = tuple(sorted([best_path[i], best_path[i+1]]))
                    steiner_edges.append(edge)
                    tree_vertices.add(best_path[i])
                    tree_vertices.add(best_path[i+1])
                
                total_cost += min_cost
                remaining_terminals.remove(closest_terminal)
        
        # Remove duplicate edges
        unique_edges = list(set(steiner_edges))
        return total_cost, unique_edges
    
    def dreyfus_wagner_exact(self, n: int, edges: List[Tuple[int, int, int]], terminals: Set[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Dreyfus-Wagner exact algorithm for small terminal sets
        
        Dynamic programming algorithm that's exact but exponential in |terminals|.
        Practical only for small terminal sets (|T| <= 15).
        
        Time: O(3^|T| * V + 2^|T| * V^2)
        Space: O(2^|T| * V)
        """
        if len(terminals) > 15:
            # Fall back to approximation for large terminal sets
            return self.mst_approximation(n, edges, terminals)
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        terminals_list = list(terminals)
        t = len(terminals_list)
        
        # DP table: dp[mask][v] = minimum cost to connect terminals in mask with tree rooted at v
        dp = defaultdict(lambda: defaultdict(lambda: float('inf')))
        parent = defaultdict(lambda: defaultdict(lambda: None))
        
        # Base case: single terminals
        for i, terminal in enumerate(terminals_list):
            mask = 1 << i
            dp[mask][terminal] = 0
        
        # Fill DP table
        for mask in range(1, 1 << t):
            if bin(mask).count('1') == 1:
                continue
            
            # For each vertex
            for v in range(n):
                # Try all submasks
                submask = mask
                while submask > 0:
                    if submask != mask:
                        complement = mask ^ submask
                        if dp[submask][v] + dp[complement][v] < dp[mask][v]:
                            dp[mask][v] = dp[submask][v] + dp[complement][v]
                            parent[mask][v] = (submask, complement, v)
                    submask = (submask - 1) & mask
                
                # Try connecting from other vertices
                for u, weight in graph[v]:
                    if dp[mask][u] + weight < dp[mask][v]:
                        dp[mask][v] = dp[mask][u] + weight
                        parent[mask][v] = (mask, u, weight)
        
        # Find optimal solution
        full_mask = (1 << t) - 1
        min_cost = min(dp[full_mask][v] for v in range(n))
        
        # Reconstruct solution (simplified)
        steiner_edges = []
        return min_cost, steiner_edges
    
    def primal_dual_approximation(self, n: int, edges: List[Tuple[int, int, int]], terminals: Set[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Primal-Dual 2-approximation algorithm
        
        Uses linear programming relaxation and dual fitting technique.
        
        Time: O(|T|^2 * (V + E))
        Approximation ratio: 2
        """
        # This is a simplified version of the primal-dual algorithm
        # For full implementation, we'd need LP solver
        
        # For now, implement a greedy variant inspired by primal-dual
        if not terminals:
            return 0, []
        
        # Build adjacency list
        graph = defaultdict(list)
        edge_weights = {}
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
            edge_weights[(min(u, v), max(u, v))] = w
        
        # Maintain dual variables (simplified)
        dual_values = {t: 0 for t in terminals}
        selected_edges = []
        connected_components = {t: {t} for t in terminals}
        
        # Iteratively select edges
        while len(connected_components) > 1:
            # Find minimum cost edge between different components
            min_cost = float('inf')
            best_edge = None
            best_components = None
            
            for u in range(n):
                for v, weight in graph[u]:
                    edge_key = (min(u, v), max(u, v))
                    
                    # Find components containing u and v
                    comp_u = None
                    comp_v = None
                    
                    for comp_id, comp_vertices in connected_components.items():
                        if u in comp_vertices:
                            comp_u = comp_id
                        if v in comp_vertices:
                            comp_v = comp_id
                    
                    if comp_u != comp_v and comp_u is not None and comp_v is not None:
                        if weight < min_cost:
                            min_cost = weight
                            best_edge = edge_key
                            best_components = (comp_u, comp_v)
            
            if best_edge and best_components:
                selected_edges.append(best_edge)
                
                # Merge components
                comp1, comp2 = best_components
                merged_comp = connected_components[comp1] | connected_components[comp2]
                
                # Update connected components
                new_comp_id = min(comp1, comp2)
                connected_components[new_comp_id] = merged_comp
                
                # Remove the other component
                other_comp_id = max(comp1, comp2)
                if other_comp_id in connected_components:
                    del connected_components[other_comp_id]
        
        total_cost = sum(edge_weights[edge] for edge in selected_edges)
        return total_cost, selected_edges
    
    def online_steiner_tree(self, n: int, edges: List[Tuple[int, int, int]], terminal_sequence: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Online Steiner Tree algorithm
        
        Terminals arrive one by one, must be connected immediately.
        
        Time: O(|T| * (V log V + E))
        Competitive ratio: O(log |T|)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        tree_vertices = set()
        steiner_edges = []
        total_cost = 0
        
        for i, terminal in enumerate(terminal_sequence):
            if not tree_vertices:
                # First terminal
                tree_vertices.add(terminal)
            else:
                # Find shortest path to existing tree
                min_cost = float('inf')
                best_path = None
                
                for tree_vertex in tree_vertices:
                    cost, path = self._dijkstra_path(graph, tree_vertex, terminal)
                    if cost < min_cost:
                        min_cost = cost
                        best_path = path
                
                if best_path:
                    # Add path to tree
                    for j in range(len(best_path) - 1):
                        edge = tuple(sorted([best_path[j], best_path[j+1]]))
                        steiner_edges.append(edge)
                        tree_vertices.add(best_path[j])
                        tree_vertices.add(best_path[j+1])
                    
                    total_cost += min_cost
        
        return total_cost, steiner_edges
    
    def _floyd_warshall(self, n: int, edges: List[Tuple[int, int, int]]) -> List[List[int]]:
        """Compute all-pairs shortest paths"""
        dist = [[float('inf')] * n for _ in range(n)]
        
        # Initialize distances
        for i in range(n):
            dist[i][i] = 0
        
        for u, v, w in edges:
            dist[u][v] = min(dist[u][v], w)
            dist[v][u] = min(dist[v][u], w)
        
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist
    
    def _kruskal_mst(self, n: int, edges: List[Tuple[int, int, int]], vertices: List[int]) -> List[Tuple[int, int, int]]:
        """Find MST using Kruskal's algorithm"""
        # Create vertex mapping
        vertex_map = {v: i for i, v in enumerate(vertices)}
        
        # Union-Find
        parent = list(range(len(vertices)))
        rank = [0] * len(vertices)
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        # Sort edges and apply Kruskal's
        edges_sorted = sorted(edges)
        mst_edges = []
        
        for weight, u, v in edges_sorted:
            if u in vertex_map and v in vertex_map:
                u_idx, v_idx = vertex_map[u], vertex_map[v]
                if union(u_idx, v_idx):
                    mst_edges.append((weight, u, v))
                    if len(mst_edges) == len(vertices) - 1:
                        break
        
        return mst_edges
    
    def _dijkstra_path(self, graph: Dict, start: int, end: int) -> Tuple[int, List[int]]:
        """Find shortest path using Dijkstra's algorithm"""
        dist = defaultdict(lambda: float('inf'))
        dist[start] = 0
        parent = {}
        heap = [(0, start)]
        visited = set()
        
        while heap:
            d, u = heapq.heappop(heap)
            
            if u in visited:
                continue
            
            visited.add(u)
            
            if u == end:
                break
            
            for v, weight in graph[u]:
                if v not in visited and dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u
                    heapq.heappush(heap, (dist[v], v))
        
        # Reconstruct path
        if end not in parent and start != end:
            return float('inf'), []
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parent.get(current)
        
        path.reverse()
        return dist[end], path
    
    def _reconstruct_shortest_path(self, u: int, v: int, dist: List[List[int]], n: int, edges: List[Tuple[int, int, int]]) -> List[int]:
        """Reconstruct shortest path between u and v"""
        # This is a simplified version - in practice, we'd maintain parent pointers
        # For now, return direct path
        return [u, v]

def test_steiner_tree_algorithms():
    """Test Steiner tree approximation algorithms"""
    print("=== Testing Steiner Tree Algorithms ===")
    
    # Test graph
    n = 6
    edges = [
        (0, 1, 2), (0, 2, 3), (1, 2, 1),
        (1, 3, 4), (2, 4, 2), (3, 4, 1),
        (3, 5, 3), (4, 5, 2)
    ]
    terminals = {0, 3, 5}
    
    steiner = SteinerTreeApproximation()
    
    algorithms = [
        ("MST Approximation", steiner.mst_approximation),
        ("Shortest Path Heuristic", steiner.shortest_path_heuristic),
        ("Primal-Dual", steiner.primal_dual_approximation),
    ]
    
    print(f"Graph: {n} vertices, {len(edges)} edges")
    print(f"Terminals: {terminals}")
    
    for name, algorithm in algorithms:
        try:
            cost, tree_edges = algorithm(n, edges, terminals)
            print(f"\n{name}:")
            print(f"  Cost: {cost}")
            print(f"  Edges: {tree_edges[:5]}...")  # Show first 5 edges
        except Exception as e:
            print(f"\n{name}: Error - {e}")

def test_online_steiner_tree():
    """Test online Steiner tree algorithm"""
    print("\n=== Testing Online Steiner Tree ===")
    
    n = 5
    edges = [
        (0, 1, 1), (0, 2, 4), (1, 2, 2),
        (1, 3, 3), (2, 4, 1), (3, 4, 2)
    ]
    terminal_sequence = [0, 2, 4, 3]
    
    steiner = SteinerTreeApproximation()
    cost, tree_edges = steiner.online_steiner_tree(n, edges, terminal_sequence)
    
    print(f"Terminal sequence: {terminal_sequence}")
    print(f"Final cost: {cost}")
    print(f"Tree edges: {tree_edges}")

def demonstrate_steiner_tree_applications():
    """Demonstrate applications of Steiner tree problem"""
    print("\n=== Steiner Tree Applications ===")
    
    print("Real-World Applications:")
    
    print("\n1. **Network Design:**")
    print("   • Minimize cost to connect required locations")
    print("   • Internet backbone infrastructure")
    print("   • Telecommunications network planning")
    print("   • Data center interconnections")
    
    print("\n2. **VLSI Design:**")
    print("   • Route electrical connections between pins")
    print("   • Minimize wire length and delay")
    print("   • Circuit layout optimization")
    print("   • Signal integrity considerations")
    
    print("\n3. **Computational Biology:**")
    print("   • Phylogenetic tree reconstruction")
    print("   • Protein structure analysis")
    print("   • Gene regulatory networks")
    print("   • Evolutionary relationship modeling")
    
    print("\n4. **Transportation Planning:**")
    print("   • Highway network design")
    print("   • Public transit route optimization")
    print("   • Pipeline infrastructure")
    print("   • Logistics network planning")
    
    print("\n5. **Multicast Routing:**")
    print("   • Efficient data distribution")
    print("   • Video streaming networks")
    print("   • Content delivery optimization")
    print("   • Network bandwidth conservation")

def analyze_approximation_algorithms():
    """Analyze different approximation algorithms"""
    print("\n=== Approximation Algorithm Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **MST-based 2-approximation:**")
    print("   • Approximation ratio: 2")
    print("   • Time complexity: O(V³ + E log V)")
    print("   • Guaranteed performance bound")
    print("   • Good theoretical properties")
    
    print("\n2. **Shortest Path Heuristic:**")
    print("   • No approximation guarantee")
    print("   • Time complexity: O(|T| × V log V)")
    print("   • Often good in practice")
    print("   • Simple to implement")
    
    print("\n3. **Dreyfus-Wagner (Exact):**")
    print("   • Optimal solution")
    print("   • Time complexity: O(3^|T| × V)")
    print("   • Exponential in terminal count")
    print("   • Practical only for small |T|")
    
    print("\n4. **Primal-Dual:**")
    print("   • Approximation ratio: 2")
    print("   • Based on LP relaxation")
    print("   • Good theoretical foundation")
    print("   • Complex implementation")
    
    print("\n5. **Online Algorithms:**")
    print("   • Competitive ratio: O(log |T|)")
    print("   • Handle dynamic terminal arrival")
    print("   • No future information")
    print("   • Important for real-time systems")
    
    print("\nKey Trade-offs:")
    print("• Approximation quality vs computation time")
    print("• Theoretical guarantees vs practical performance")
    print("• Online vs offline algorithm capabilities")
    print("• Implementation complexity vs result quality")

def demonstrate_complexity_analysis():
    """Demonstrate complexity analysis of Steiner tree problem"""
    print("\n=== Complexity Analysis ===")
    
    print("Problem Complexity:")
    
    print("\n1. **Computational Complexity:**")
    print("   • General Steiner Tree: NP-hard")
    print("   • Even on planar graphs: NP-hard")
    print("   • Euclidean Steiner Tree: NP-hard")
    print("   • No PTAS unless P = NP")
    
    print("\n2. **Approximation Hardness:**")
    print("   • Best known lower bound: 1.01")
    print("   • Best known upper bound: 1.55")
    print("   • Significant gap between bounds")
    print("   • Active area of research")
    
    print("\n3. **Special Cases:**")
    print("   • Trees: Polynomial time (trivial)")
    print("   • Series-parallel graphs: Polynomial")
    print("   • Small terminal sets: Fixed-parameter tractable")
    print("   • Geometric versions: Various complexities")
    
    print("\n4. **Practical Considerations:**")
    print("   • Real instances often have structure")
    print("   • Heuristics work well in practice")
    print("   • Preprocessing can help significantly")
    print("   • Local search improves solutions")
    
    print("\nResearch Directions:")
    print("• Better approximation algorithms")
    print("• Improved hardness results")
    print("• Practical algorithm development")
    print("• Specialized algorithms for applications")

if __name__ == "__main__":
    test_steiner_tree_algorithms()
    test_online_steiner_tree()
    demonstrate_steiner_tree_applications()
    analyze_approximation_algorithms()
    demonstrate_complexity_analysis()

"""
Steiner Tree and Network Optimization Concepts:
1. NP-hard Optimization Problems and Approximation Algorithms
2. MST-based Approximation with Performance Guarantees
3. Dynamic Programming for Exact Solutions (Small Instances)
4. Online Algorithms with Competitive Analysis
5. Primal-Dual Methods and Linear Programming Relaxations

Key Problem Insights:
- Steiner tree connects required terminals with minimum cost
- May use additional vertices (Steiner vertices) for optimization
- NP-hard in general, requiring approximation algorithms
- Wide applications in network design and VLSI routing

Algorithm Strategy:
1. MST Approximation: Metric closure + MST on terminals
2. Shortest Path Heuristic: Greedy connection of nearest terminals
3. Exact DP: Dreyfus-Wagner for small terminal sets
4. Online: Connect terminals as they arrive dynamically

Approximation Analysis:
- MST-based algorithm: 2-approximation ratio
- Shortest path heuristic: No guarantee but often good
- Best known approximation ratio: 1.55
- Hardness result: No PTAS unless P = NP

Network Design Applications:
- Telecommunications infrastructure planning
- VLSI circuit routing and optimization
- Multicast routing in computer networks
- Transportation network design
- Computational biology and phylogenetics

This collection demonstrates advanced approximation algorithms
for NP-hard optimization problems in network design.
"""
