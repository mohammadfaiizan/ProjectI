"""
684. Redundant Connection
Difficulty: Medium

Problem:
In this problem, a tree is an undirected graph that is connected and has no cycles.

You are given a graph that started as a tree with n nodes labeled from 1 to n, with one 
additional edge added. The added edge has two different vertices chosen from 1 to n, and 
was not an edge that already existed. The graph is represented as an array edges where 
edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the graph.

Return an edge that can be removed so that the resulting graph is a tree of n nodes. 
If there are multiple answers, return the answer that occurs last in the input.

Examples:
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]

Input: edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
Output: [1,4]

Constraints:
- n == edges.length
- 3 <= n <= 1000
- edges[i].length == 2
- 1 <= ai < bi <= n
- ai != bi
- There are no repeated edges
"""

from typing import List

class UnionFind:
    """Union-Find with path compression and union by rank"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank, returns True if union performed"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already connected, would create cycle
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two nodes are connected"""
        return self.find(x) == self.find(y)

class Solution:
    def findRedundantConnection_approach1_union_find(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: Union-Find (Optimal)
        
        Process edges one by one. The first edge that connects already 
        connected components creates a cycle.
        
        Time: O(N * α(N)) ≈ O(N)
        Space: O(N)
        """
        n = len(edges)
        uf = UnionFind(n + 1)  # 1-indexed
        
        for edge in edges:
            u, v = edge
            
            # If already connected, this edge creates a cycle
            if uf.connected(u, v):
                return edge
            
            # Union the components
            uf.union(u, v)
        
        return []  # Should never reach here given problem constraints
    
    def findRedundantConnection_approach2_dfs_cycle_detection(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: DFS Cycle Detection
        
        Build graph incrementally, check for cycles using DFS.
        
        Time: O(N^2) - O(N) for each edge check
        Space: O(N)
        """
        from collections import defaultdict
        
        graph = defaultdict(list)
        
        def has_cycle(start, target, visited):
            """Check if there's a path from start to target"""
            if start == target:
                return True
            
            visited.add(start)
            
            for neighbor in graph[start]:
                if neighbor not in visited:
                    if has_cycle(neighbor, target, visited):
                        return True
            
            return False
        
        for u, v in edges:
            # Check if adding this edge creates a cycle
            if u in graph and v in graph and has_cycle(u, v, set()):
                return [u, v]
            
            # Add edge to graph
            graph[u].append(v)
            graph[v].append(u)
        
        return []
    
    def findRedundantConnection_approach3_iterative_dfs(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 3: Iterative DFS for Cycle Detection
        
        Use stack-based DFS to avoid recursion.
        
        Time: O(N^2)
        Space: O(N)
        """
        from collections import defaultdict
        
        graph = defaultdict(set)
        
        def path_exists(start, end):
            """Check if path exists using iterative DFS"""
            if start == end:
                return True
            
            stack = [start]
            visited = {start}
            
            while stack:
                node = stack.pop()
                
                for neighbor in graph[node]:
                    if neighbor == end:
                        return True
                    
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            return False
        
        for u, v in edges:
            if path_exists(u, v):
                return [u, v]
            
            graph[u].add(v)
            graph[v].add(u)
        
        return []
    
    def findRedundantConnection_approach4_parent_tracking(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 4: Simple Parent Tracking
        
        Use simple parent array without rank optimization.
        
        Time: O(N^2) worst case without path compression
        Space: O(N)
        """
        n = len(edges)
        parent = list(range(n + 1))
        
        def find(x):
            """Simple find without path compression"""
            while parent[x] != x:
                x = parent[x]
            return x
        
        def union(x, y):
            """Simple union without rank"""
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return False
            
            parent[root_x] = root_y
            return True
        
        for edge in edges:
            u, v = edge
            
            if not union(u, v):
                return edge
        
        return []

def test_redundant_connection():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (edges, expected)
        ([[1,2],[1,3],[2,3]], [2,3]),
        ([[1,2],[2,3],[3,4],[1,4],[1,5]], [1,4]),
        ([[1,2],[2,3],[3,1]], [3,1]),
        ([[1,2],[2,3],[3,4],[4,1]], [4,1]),
        ([[2,3],[5,2],[1,5],[4,3],[4,1]], [4,1]),
    ]
    
    approaches = [
        ("Union-Find", solution.findRedundantConnection_approach1_union_find),
        ("DFS Cycle Detection", solution.findRedundantConnection_approach2_dfs_cycle_detection),
        ("Iterative DFS", solution.findRedundantConnection_approach3_iterative_dfs),
        ("Parent Tracking", solution.findRedundantConnection_approach4_parent_tracking),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (edges, expected) in enumerate(test_cases):
            result = func(edges[:])  # Copy to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Edges: {edges}, Expected: {expected}, Got: {result}")

def demonstrate_cycle_detection():
    """Demonstrate cycle detection process"""
    print("\n=== Cycle Detection Demo ===")
    
    edges = [[1,2],[1,3],[2,3]]
    print(f"Edges: {edges}")
    print(f"Processing edges one by one:")
    
    uf = UnionFind(4)  # 0-3, using 1-3
    
    for i, (u, v) in enumerate(edges):
        print(f"\nStep {i+1}: Processing edge ({u}, {v})")
        
        # Check if already connected
        if uf.connected(u, v):
            print(f"  ❌ Nodes {u} and {v} are already connected!")
            print(f"  This edge creates a cycle.")
            print(f"  Redundant edge: [{u}, {v}]")
            break
        else:
            print(f"  ✓ Nodes {u} and {v} are not connected.")
            print(f"  Union({u}, {v})")
            uf.union(u, v)
            
            # Show current component structure
            components = {}
            for node in [1, 2, 3]:
                root = uf.find(node)
                if root not in components:
                    components[root] = []
                components[root].append(node)
            
            print(f"  Current components: {list(components.values())}")

def analyze_tree_properties():
    """Analyze tree properties and redundant connections"""
    print("\n=== Tree Properties Analysis ===")
    
    print("Tree Characteristics:")
    print("• **Connected:** Every node reachable from every other node")
    print("• **Acyclic:** No cycles in the graph")
    print("• **Minimal:** Removing any edge disconnects the graph")
    print("• **Edges:** Exactly n-1 edges for n nodes")
    
    print("\nRedundant Connection Problem:")
    print("• Start with valid tree (n-1 edges)")
    print("• Add one extra edge (total n edges)")
    print("• Extra edge creates exactly one cycle")
    print("• Goal: Find which edge to remove")
    
    print("\nWhy Union-Find Works:")
    print("1. **Process edges in order**")
    print("2. **Track connected components**")
    print("3. **First edge connecting same component = cycle**")
    print("4. **Return immediately (last occurrence)**")
    
    print("\nUnion-Find Advantages:")
    print("• O(N) time complexity with optimizations")
    print("• O(N) space complexity")
    print("• Natural fit for connectivity problems")
    print("• Handles dynamic updates efficiently")
    
    print("\nAlternative Approaches:")
    print("• **DFS Cycle Detection:** O(N²) time")
    print("• **BFS Cycle Detection:** O(N²) time")
    print("• **Incremental Graph Building:** O(N²) time")
    print("• **Spanning Tree Algorithms:** Overkill")

def demonstrate_edge_order_importance():
    """Demonstrate importance of edge processing order"""
    print("\n=== Edge Order Importance Demo ===")
    
    edges = [[1,2],[2,3],[3,4],[1,4],[1,5]]
    print(f"Edges: {edges}")
    print(f"Expected result: [1,4] (last edge that creates cycle)")
    
    print(f"\nStep-by-step Union-Find process:")
    
    uf = UnionFind(6)  # 0-5, using 1-5
    
    for i, (u, v) in enumerate(edges):
        print(f"\nStep {i+1}: Edge ({u}, {v})")
        
        # Show current components before processing
        components = {}
        for node in range(1, 6):
            root = uf.find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)
        
        existing_components = [comp for comp in components.values() if len(comp) > 1]
        if existing_components:
            print(f"  Current components: {existing_components}")
        else:
            print(f"  Current components: All nodes isolated")
        
        if uf.connected(u, v):
            print(f"  🔴 CYCLE DETECTED! Nodes {u} and {v} already connected")
            print(f"  Redundant edge: [{u}, {v}]")
            break
        else:
            print(f"  ✅ Union({u}, {v}) - No cycle")
            uf.union(u, v)
    
    print(f"\nWhy edge order matters:")
    print(f"• We want the LAST edge that creates a cycle")
    print(f"• Multiple edges could create cycles")
    print(f"• Problem asks for last occurrence in input")
    print(f"• Union-Find naturally finds first cycle-creating edge")

def compare_cycle_detection_methods():
    """Compare different cycle detection methods"""
    print("\n=== Cycle Detection Methods Comparison ===")
    
    print("1. **Union-Find Approach:**")
    print("   ✅ Optimal O(N) time complexity")
    print("   ✅ Simple implementation")
    print("   ✅ Handles dynamic connectivity")
    print("   ✅ Natural fit for the problem")
    print("   ❌ Requires Union-Find knowledge")
    
    print("\n2. **DFS-Based Detection:**")
    print("   ✅ Intuitive graph traversal")
    print("   ✅ Educational value")
    print("   ✅ Direct cycle detection")
    print("   ❌ O(N²) time complexity")
    print("   ❌ Rebuilds graph for each edge")
    
    print("\n3. **Incremental Graph Building:**")
    print("   ✅ Clear step-by-step process")
    print("   ✅ Easy to understand")
    print("   ✅ Explicit graph representation")
    print("   ❌ O(N²) time complexity")
    print("   ❌ Extra space for adjacency lists")
    
    print("\nReal-world Applications:")
    print("• **Network Design:** Preventing redundant connections")
    print("• **Circuit Analysis:** Identifying feedback loops")
    print("• **Dependency Resolution:** Circular dependency detection")
    print("• **Social Networks:** Community structure analysis")
    print("• **Transportation:** Route optimization")
    
    print("\nKey Insights:")
    print("• Trees have exactly n-1 edges for n nodes")
    print("• Adding one edge creates exactly one cycle")
    print("• Union-Find efficiently tracks connectivity")
    print("• First cycle-creating edge is the answer")
    print("• Edge processing order determines result")

if __name__ == "__main__":
    test_redundant_connection()
    demonstrate_cycle_detection()
    analyze_tree_properties()
    demonstrate_edge_order_importance()
    compare_cycle_detection_methods()

"""
Union-Find Concepts:
1. Cycle Detection in Undirected Graphs
2. Dynamic Connectivity Queries
3. Tree Property Maintenance
4. Incremental Graph Processing

Key Problem Insights:
- Tree + 1 edge = exactly one cycle
- Union-Find detects first cycle-creating edge
- Edge processing order determines result
- Redundant edge breaks tree property

Algorithm Strategy:
1. Process edges in input order
2. Use Union-Find to track components
3. First edge connecting same component = cycle
4. Return immediately for "last occurrence"

Union-Find Optimizations:
- Path compression: flattens tree structure
- Union by rank: balances component trees
- Combined: O(α(N)) amortized per operation
- Total time: O(N) for N edges

Real-world Applications:
- Network topology design
- Circuit analysis and validation
- Dependency graph management
- Social network analysis
- Transportation network optimization

This problem demonstrates Union-Find for
dynamic cycle detection in graph construction.
"""
