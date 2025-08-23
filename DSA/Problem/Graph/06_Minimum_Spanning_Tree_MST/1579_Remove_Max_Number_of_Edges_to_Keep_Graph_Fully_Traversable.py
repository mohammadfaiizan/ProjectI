"""
1579. Remove Max Number of Edges to Keep Graph Fully Traversable
Difficulty: Hard

Problem:
Alice and Bob have an undirected graph of n nodes and 3 types of edges:
- Type 1: Can be traversed by Alice only.
- Type 2: Can be traversed by Bob only.
- Type 3: Can be traversed by both Alice and Bob.

Given an array edges where edges[i] = [typei, ui, vi] represents a bidirectional edge 
of type typei between nodes ui and vi, find the maximum number of edges you can remove 
so that after removing them, the graph remains fully traversable for both Alice and Bob. 
A graph is fully traversable for Alice (and Bob) if Alice (and Bob) can reach every other 
node from any node.

Return the maximum number of edges you can remove, or return -1 if it's impossible to 
make the graph fully traversable for both Alice and Bob.

Examples:
Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,4],[2,3,4]]
Output: 2

Input: n = 4, edges = [[3,1,2],[3,2,3],[1,1,4],[2,1,4]]
Output: 0

Input: n = 4, edges = [[3,2,3],[1,1,2],[2,3,4]]
Output: -1

Constraints:
- 1 <= n <= 10^5
- 1 <= edges.length <= min(10^5, 3 * n * (n-1) / 2)
- edges[i].length == 3
- 1 <= typei <= 3
- 1 <= ui < vi <= n
- All tuples (typei, ui, vi) are distinct
"""

from typing import List

class Solution:
    def maxNumEdgesToRemove_approach1_union_find_mst(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 1: Union-Find with MST Strategy (Optimal)
        
        Use Union-Find to build minimum spanning forests for Alice and Bob separately,
        prioritizing type 3 edges that benefit both.
        
        Time: O(E * α(V)) where α is inverse Ackermann function
        Space: O(V)
        """
        # Separate edges by type
        type1_edges = []  # Alice only
        type2_edges = []  # Bob only
        type3_edges = []  # Both Alice and Bob
        
        for edge_type, u, v in edges:
            if edge_type == 1:
                type1_edges.append((u, v))
            elif edge_type == 2:
                type2_edges.append((u, v))
            else:  # edge_type == 3
                type3_edges.append((u, v))
        
        # Union-Find for Alice and Bob
        alice_uf = UnionFind(n)
        bob_uf = UnionFind(n)
        
        edges_used = 0
        
        # Step 1: Add type 3 edges (beneficial for both)
        for u, v in type3_edges:
            alice_connected = alice_uf.union(u, v)
            bob_connected = bob_uf.union(u, v)
            
            # If edge helps at least one person, keep it
            if alice_connected or bob_connected:
                edges_used += 1
        
        # Step 2: Add type 1 edges for Alice
        for u, v in type1_edges:
            if alice_uf.union(u, v):
                edges_used += 1
        
        # Step 3: Add type 2 edges for Bob
        for u, v in type2_edges:
            if bob_uf.union(u, v):
                edges_used += 1
        
        # Check if both Alice and Bob can traverse the entire graph
        if alice_uf.components == 1 and bob_uf.components == 1:
            return len(edges) - edges_used
        else:
            return -1
    
    def maxNumEdgesToRemove_approach2_greedy_mst_construction(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 2: Greedy MST Construction
        
        Greedily construct MSTs for both players, prioritizing shared edges.
        
        Time: O(E * α(V))
        Space: O(V)
        """
        # Create separate Union-Find structures
        alice_uf = UnionFind(n)
        bob_uf = UnionFind(n)
        
        # Process edges in order of preference: type 3 first (shared benefit)
        sorted_edges = []
        
        # Add type 3 edges first (highest priority)
        for edge_type, u, v in edges:
            if edge_type == 3:
                sorted_edges.append((3, u, v))
        
        # Add type 1 and type 2 edges
        for edge_type, u, v in edges:
            if edge_type != 3:
                sorted_edges.append((edge_type, u, v))
        
        edges_used = 0
        
        for edge_type, u, v in sorted_edges:
            alice_can_use = (edge_type == 1 or edge_type == 3)
            bob_can_use = (edge_type == 2 or edge_type == 3)
            
            alice_needs = alice_can_use and not alice_uf.connected(u, v)
            bob_needs = bob_can_use and not bob_uf.connected(u, v)
            
            if alice_needs or bob_needs:
                # At least one person benefits from this edge
                if alice_can_use:
                    alice_uf.union(u, v)
                if bob_can_use:
                    bob_uf.union(u, v)
                edges_used += 1
        
        # Verify full traversability
        if alice_uf.components == 1 and bob_uf.components == 1:
            return len(edges) - edges_used
        else:
            return -1
    
    def maxNumEdgesToRemove_approach3_kruskal_variant(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 3: Kruskal's Algorithm Variant
        
        Modified Kruskal's algorithm for dual-MST construction.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        # Sort edges to prioritize type 3 (shared edges)
        sorted_edges = sorted(edges, key=lambda x: (0 if x[0] == 3 else x[0]))
        
        alice_uf = UnionFind(n)
        bob_uf = UnionFind(n)
        essential_edges = 0
        
        for edge_type, u, v in sorted_edges:
            alice_benefits = False
            bob_benefits = False
            
            # Check if Alice can use and benefits from this edge
            if edge_type in [1, 3] and not alice_uf.connected(u, v):
                alice_benefits = True
            
            # Check if Bob can use and benefits from this edge
            if edge_type in [2, 3] and not bob_uf.connected(u, v):
                bob_benefits = True
            
            # Add edge if it benefits at least one person
            if alice_benefits or bob_benefits:
                if edge_type in [1, 3]:
                    alice_uf.union(u, v)
                if edge_type in [2, 3]:
                    bob_uf.union(u, v)
                essential_edges += 1
        
        # Check connectivity
        if alice_uf.components == 1 and bob_uf.components == 1:
            return len(edges) - essential_edges
        else:
            return -1
    
    def maxNumEdgesToRemove_approach4_incremental_mst(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 4: Incremental MST Construction
        
        Build MSTs incrementally, tracking redundant edges.
        
        Time: O(E * α(V))
        Space: O(V)
        """
        # Group edges by type
        edge_groups = {1: [], 2: [], 3: []}
        for edge_type, u, v in edges:
            edge_groups[edge_type].append((u, v))
        
        # Initialize Union-Find structures
        alice_uf = UnionFind(n)
        bob_uf = UnionFind(n)
        
        essential_count = 0
        
        # Phase 1: Add type 3 edges (most valuable)
        for u, v in edge_groups[3]:
            alice_connected_before = alice_uf.connected(u, v)
            bob_connected_before = bob_uf.connected(u, v)
            
            # Add edge if it helps either Alice or Bob
            if not alice_connected_before or not bob_connected_before:
                alice_uf.union(u, v)
                bob_uf.union(u, v)
                essential_count += 1
        
        # Phase 2: Add type 1 edges for Alice
        for u, v in edge_groups[1]:
            if alice_uf.union(u, v):
                essential_count += 1
        
        # Phase 3: Add type 2 edges for Bob
        for u, v in edge_groups[2]:
            if bob_uf.union(u, v):
                essential_count += 1
        
        # Verify full connectivity
        alice_connected = (alice_uf.components == 1)
        bob_connected = (bob_uf.components == 1)
        
        if alice_connected and bob_connected:
            return len(edges) - essential_count
        else:
            return -1
    
    def maxNumEdgesToRemove_approach5_optimal_edge_selection(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 5: Optimal Edge Selection Strategy
        
        Sophisticated edge selection with lookahead optimization.
        
        Time: O(E * α(V))
        Space: O(V + E)
        """
        # Create edge priority: type 3 > type 1/2
        prioritized_edges = []
        
        # Type 3 edges first (can help both players)
        for i, (edge_type, u, v) in enumerate(edges):
            if edge_type == 3:
                prioritized_edges.append((0, edge_type, u, v, i))
        
        # Type 1 and 2 edges
        for i, (edge_type, u, v) in enumerate(edges):
            if edge_type != 3:
                prioritized_edges.append((1, edge_type, u, v, i))
        
        # Sort by priority
        prioritized_edges.sort()
        
        alice_uf = UnionFind(n)
        bob_uf = UnionFind(n)
        used_edges = set()
        
        for priority, edge_type, u, v, original_index in prioritized_edges:
            alice_can_use = edge_type in [1, 3]
            bob_can_use = edge_type in [2, 3]
            
            alice_needs = alice_can_use and not alice_uf.connected(u, v)
            bob_needs = bob_can_use and not bob_uf.connected(u, v)
            
            if alice_needs or bob_needs:
                # This edge is essential for at least one player
                used_edges.add(original_index)
                
                if alice_can_use:
                    alice_uf.union(u, v)
                if bob_can_use:
                    bob_uf.union(u, v)
        
        # Check if both graphs are fully connected
        if alice_uf.components == 1 and bob_uf.components == 1:
            return len(edges) - len(used_edges)
        else:
            return -1

class UnionFind:
    """Optimized Union-Find with path compression and union by rank"""
    
    def __init__(self, n: int):
        self.parent = list(range(n + 1))  # 1-indexed
        self.rank = [0] * (n + 1)
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
    
    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

def test_max_edges_to_remove():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, expected)
        (4, [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,4],[2,3,4]], 2),
        (4, [[3,1,2],[3,2,3],[1,1,4],[2,1,4]], 0),
        (4, [[3,2,3],[1,1,2],[2,3,4]], -1),
        (5, [[3,1,2],[3,2,3],[3,3,4],[1,4,5],[2,1,5]], 1),
        (2, [[1,1,2],[2,1,2],[3,1,2]], 2),
        (3, [[3,1,2],[3,1,3],[1,2,3]], 0),
    ]
    
    approaches = [
        ("Union-Find MST", solution.maxNumEdgesToRemove_approach1_union_find_mst),
        ("Greedy MST Construction", solution.maxNumEdgesToRemove_approach2_greedy_mst_construction),
        ("Kruskal Variant", solution.maxNumEdgesToRemove_approach3_kruskal_variant),
        ("Incremental MST", solution.maxNumEdgesToRemove_approach4_incremental_mst),
        ("Optimal Edge Selection", solution.maxNumEdgesToRemove_approach5_optimal_edge_selection),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, expected) in enumerate(test_cases):
            result = func(n, edges[:])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={expected}, got={result}")

def demonstrate_dual_mst_concept():
    """Demonstrate dual MST construction concept"""
    print("\n=== Dual MST Construction Demo ===")
    
    n = 4
    edges = [[3,1,2],[3,2,3],[1,1,3],[1,2,4],[1,1,4],[2,3,4]]
    
    print(f"Graph: {n} nodes")
    print(f"Edges: {edges}")
    print(f"Edge types: 1=Alice only, 2=Bob only, 3=Both")
    
    # Separate edges by type
    alice_edges = []
    bob_edges = []
    
    for edge_type, u, v in edges:
        if edge_type == 1:
            alice_edges.append((u, v, "Alice only"))
        elif edge_type == 2:
            bob_edges.append((u, v, "Bob only"))
        else:  # edge_type == 3
            alice_edges.append((u, v, "Shared"))
            bob_edges.append((u, v, "Shared"))
    
    print(f"\nAlice's available edges: {alice_edges}")
    print(f"Bob's available edges: {bob_edges}")
    
    # Build MSTs
    print(f"\nBuilding MSTs with shared edges prioritized:")
    
    alice_uf = UnionFind(n)
    bob_uf = UnionFind(n)
    used_edges = []
    
    # Process in order: shared edges first
    edge_processing_order = []
    for i, (edge_type, u, v) in enumerate(edges):
        priority = 0 if edge_type == 3 else edge_type
        edge_processing_order.append((priority, edge_type, u, v, i))
    
    edge_processing_order.sort()
    
    for priority, edge_type, u, v, original_idx in edge_processing_order:
        alice_can_use = edge_type in [1, 3]
        bob_can_use = edge_type in [2, 3]
        
        alice_needs = alice_can_use and not alice_uf.connected(u, v)
        bob_needs = bob_can_use and not bob_uf.connected(u, v)
        
        type_name = ["", "Alice only", "Bob only", "Shared"][edge_type]
        
        print(f"\nConsider edge {original_idx}: ({u},{v}) - {type_name}")
        print(f"  Alice needs: {alice_needs}, Bob needs: {bob_needs}")
        
        if alice_needs or bob_needs:
            used_edges.append(original_idx)
            if alice_can_use:
                alice_uf.union(u, v)
            if bob_can_use:
                bob_uf.union(u, v)
            print(f"  ✓ ESSENTIAL - used by at least one player")
        else:
            print(f"  ✗ REDUNDANT - can be removed")
    
    print(f"\nResults:")
    print(f"  Alice connected: {alice_uf.components == 1}")
    print(f"  Bob connected: {bob_uf.components == 1}")
    print(f"  Essential edges: {used_edges}")
    print(f"  Removable edges: {len(edges) - len(used_edges)}")

def demonstrate_edge_priority_strategy():
    """Demonstrate edge prioritization strategy"""
    print("\n=== Edge Priority Strategy Demo ===")
    
    print("Edge Processing Priority Strategy:")
    
    print("\n1. **Type 3 Edges (Shared) - Highest Priority:**")
    print("   • Benefits both Alice and Bob")
    print("   • Most efficient use of edges")
    print("   • Reduces total edges needed")
    print("   • Process first to maximize sharing")
    
    print("\n2. **Type 1 Edges (Alice Only) - Medium Priority:**")
    print("   • Only helps Alice's connectivity")
    print("   • Process after shared edges")
    print("   • Fill gaps in Alice's MST")
    
    print("\n3. **Type 2 Edges (Bob Only) - Medium Priority:**")
    print("   • Only helps Bob's connectivity")
    print("   • Process after shared edges")
    print("   • Fill gaps in Bob's MST")
    
    print("\nOptimization Strategy:")
    print("• **Greedy Approach:** Always prefer edges that help more players")
    print("• **Shared First:** Type 3 edges provide maximum benefit")
    print("• **Union-Find:** Efficient connectivity testing")
    print("• **Early Termination:** Stop when both players are connected")
    
    print("\nExample Priority Ordering:")
    edges = [[3,1,2],[1,2,3],[2,3,4],[3,4,1]]
    print(f"Original edges: {edges}")
    
    prioritized = []
    for i, (edge_type, u, v) in enumerate(edges):
        priority = 0 if edge_type == 3 else edge_type
        prioritized.append((priority, edge_type, u, v))
    
    prioritized.sort()
    print(f"Prioritized order: {[(t,u,v) for p,t,u,v in prioritized]}")

def analyze_dual_mst_complexity():
    """Analyze complexity of dual MST algorithms"""
    print("\n=== Dual MST Complexity Analysis ===")
    
    print("Problem Characteristics:")
    print("• Two separate connectivity requirements")
    print("• Shared edges benefit both players")
    print("• Goal: minimize total edges while maintaining connectivity")
    print("• NP-hard in general case, but polynomial for this variant")
    
    print("\nAlgorithm Complexity:")
    
    print("\n**Time Complexity:**")
    print("• Union-Find operations: O(E * α(V))")
    print("• Edge sorting (if needed): O(E log E)")
    print("• Connectivity testing: O(1) amortized per edge")
    print("• Total: O(E * α(V)) where α is inverse Ackermann")
    
    print("\n**Space Complexity:**")
    print("• Two Union-Find structures: O(V)")
    print("• Edge storage: O(E)")
    print("• Auxiliary data structures: O(E)")
    print("• Total: O(V + E)")
    
    print("\nOptimizations:")
    print("• **Path Compression:** Accelerates Union-Find find operations")
    print("• **Union by Rank:** Balances Union-Find tree structure")
    print("• **Early Termination:** Stop when both players connected")
    print("• **Priority Processing:** Shared edges first for efficiency")
    
    print("\nPractical Performance:")
    print("• Typical case: O(E) due to nearly constant α(V)")
    print("• Worst case: O(E log* V) for extremely large graphs")
    print("• Memory efficient: Linear space complexity")
    print("• Cache friendly: Good locality in Union-Find operations")

def analyze_applications_and_extensions():
    """Analyze applications and extensions of dual connectivity"""
    print("\n=== Applications and Extensions ===")
    
    print("Real-World Applications:")
    
    print("\n1. **Network Access Control:**")
    print("   • Different user groups with different permissions")
    print("   • Shared infrastructure vs dedicated connections")
    print("   • Minimize total network infrastructure cost")
    
    print("\n2. **Transportation Systems:**")
    print("   • Different vehicle types (cars, trucks, buses)")
    print("   • Shared roads vs dedicated lanes")
    print("   • Optimize road network construction costs")
    
    print("\n3. **Multi-Tenant Cloud Networks:**")
    print("   • Different tenants with different access rights")
    print("   • Shared vs isolated network segments")
    print("   • Minimize network hardware and maintenance costs")
    
    print("\n4. **Supply Chain Networks:**")
    print("   • Different product categories")
    print("   • Shared vs dedicated distribution channels")
    print("   • Optimize logistics infrastructure")
    
    print("\nProblem Extensions:")
    
    print("\n1. **Multi-Player Traversability:**")
    print("   • Extend to more than 2 players")
    print("   • Complex edge sharing patterns")
    print("   • Generalized MST construction")
    
    print("\n2. **Weighted Edge Removal:**")
    print("   • Different costs for removing different edges")
    print("   • Weighted optimization objective")
    print("   • More complex cost-benefit analysis")
    
    print("\n3. **Dynamic Edge Addition/Removal:**")
    print("   • Online algorithm for changing edge sets")
    print("   • Incremental MST maintenance")
    print("   • Real-time network optimization")
    
    print("\n4. **Fault-Tolerant Requirements:**")
    print("   • k-connected graphs for robustness")
    print("   • Multiple edge-disjoint paths")
    print("   • Reliability-constrained optimization")
    
    print("\nTechnical Insights:")
    print("• Dual connectivity generalizes single MST problems")
    print("• Shared resources create interesting optimization trade-offs")
    print("• Union-Find provides efficient incremental connectivity")
    print("• Greedy algorithms often optimal for structured problems")
    print("• Priority-based processing crucial for performance")

if __name__ == "__main__":
    test_max_edges_to_remove()
    demonstrate_dual_mst_concept()
    demonstrate_edge_priority_strategy()
    analyze_dual_mst_complexity()
    analyze_applications_and_extensions()

"""
Dual MST and Edge Removal Concepts:
1. Dual Minimum Spanning Tree Construction
2. Shared Edge Optimization Strategies
3. Union-Find for Multi-Player Connectivity
4. Greedy Edge Selection and Prioritization
5. Network Access Control and Resource Sharing

Key Problem Insights:
- Two players need separate connectivity requirements
- Type 3 edges benefit both players (highest value)
- Goal: maximize removable edges while maintaining connectivity
- Greedy strategy: prioritize shared edges for efficiency

Algorithm Strategy:
1. Prioritize type 3 (shared) edges for maximum benefit
2. Use separate Union-Find structures for Alice and Bob
3. Add edges only if they help at least one player
4. Verify final connectivity for both players

Edge Classification and Priority:
- Type 3 (shared): Highest priority, benefits both players
- Type 1 (Alice only): Medium priority, fills Alice's gaps
- Type 2 (Bob only): Medium priority, fills Bob's gaps
- Process in priority order for optimal edge utilization

Union-Find Optimizations:
- Path compression for efficient find operations
- Union by rank for balanced tree structures
- Separate structures for independent connectivity tracking
- Component counting for connectivity verification

Dual Connectivity Requirements:
- Both players must reach all nodes
- Shared edges reduce total infrastructure needed
- Independent MST construction with shared resources
- Optimize total edge count while meeting constraints

Real-world Applications:
- Network access control with different user permissions
- Transportation systems with multiple vehicle types
- Multi-tenant cloud infrastructure optimization
- Supply chain networks with diverse product categories
- Shared vs dedicated resource allocation problems

This problem demonstrates advanced MST techniques
for multi-player connectivity optimization scenarios.
"""
