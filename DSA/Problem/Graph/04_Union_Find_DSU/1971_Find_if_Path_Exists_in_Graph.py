"""
1971. Find if Path Exists in Graph
Difficulty: Easy

Problem:
There is a bi-directional graph with n vertices, where each vertex is labeled from 0 to n - 1 
(inclusive). The edges in the graph are represented as a 2D integer array edges, where each 
edges[i] = [ui, vi] denotes a bi-directional edge between vertex ui and vertex vi. Every vertex 
pair is connected by at most one edge, and no vertex has an edge to itself.

You want to determine if there is a valid path that exists from vertex source to vertex 
destination.

Given edges and the integers n, source, and destination, return true if there is a valid 
path from source to destination, or false otherwise.

Examples:
Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
Output: true

Input: n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5
Output: false

Constraints:
- 1 <= n <= 2 * 10^5
- 0 <= edges.length <= 2 * 10^5
- edges[i].length == 2
- 0 <= ui, vi <= n - 1
- ui != vi
- 0 <= source, destination <= n - 1
- There are no duplicate edges
- There are no self edges
"""

from typing import List
from collections import deque, defaultdict

class UnionFind:
    """Union-Find for connectivity queries"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def connected(self, x, y):
        """Check if two nodes are connected"""
        return self.find(x) == self.find(y)

class Solution:
    def validPath_approach1_union_find(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 1: Union-Find (Optimal for Multiple Queries)
        
        Build Union-Find structure and check connectivity.
        
        Time: O(E * α(V)) ≈ O(E) for building, O(α(V)) ≈ O(1) for query
        Space: O(V)
        """
        if source == destination:
            return True
        
        uf = UnionFind(n)
        
        # Union all connected vertices
        for u, v in edges:
            uf.union(u, v)
        
        # Check if source and destination are connected
        return uf.connected(source, destination)
    
    def validPath_approach2_dfs(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 2: DFS
        
        Build adjacency list and use DFS to find path.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if source == destination:
            return True
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set()
        
        def dfs(node):
            """DFS to find destination"""
            if node == destination:
                return True
            
            visited.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            
            return False
        
        return dfs(source)
    
    def validPath_approach3_bfs(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 3: BFS
        
        Use BFS to find path from source to destination.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if source == destination:
            return True
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # BFS
        queue = deque([source])
        visited = {source}
        
        while queue:
            node = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor == destination:
                    return True
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def validPath_approach4_iterative_dfs(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 4: Iterative DFS
        
        Use stack-based DFS to avoid recursion.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if source == destination:
            return True
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Iterative DFS
        stack = [source]
        visited = {source}
        
        while stack:
            node = stack.pop()
            
            for neighbor in graph[node]:
                if neighbor == destination:
                    return True
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        
        return False
    
    def validPath_approach5_optimized_union_find(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 5: Optimized Union-Find with Early Termination
        
        Stop union operations once source and destination are connected.
        
        Time: O(E * α(V)) with potential early termination
        Space: O(V)
        """
        if source == destination:
            return True
        
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y
                return True
            return False
        
        # Process edges with early termination
        for u, v in edges:
            union(u, v)
            
            # Early termination check
            if find(source) == find(destination):
                return True
        
        return False

def test_valid_path():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, source, destination, expected)
        (3, [[0,1],[1,2],[2,0]], 0, 2, True),
        (6, [[0,1],[0,2],[3,5],[5,4],[4,3]], 0, 5, False),
        (1, [], 0, 0, True),
        (2, [[0,1]], 0, 1, True),
        (2, [[0,1]], 1, 0, True),
        (2, [], 0, 1, False),
        (10, [[4,3],[1,4],[4,8],[1,7],[6,4],[4,2],[7,4],[4,0],[0,9],[5,4]], 5, 9, True),
    ]
    
    approaches = [
        ("Union-Find", solution.validPath_approach1_union_find),
        ("DFS", solution.validPath_approach2_dfs),
        ("BFS", solution.validPath_approach3_bfs),
        ("Iterative DFS", solution.validPath_approach4_iterative_dfs),
        ("Optimized UF", solution.validPath_approach5_optimized_union_find),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, source, dest, expected) in enumerate(test_cases):
            result = func(n, edges[:], source, dest)  # Copy edges to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, source={source}, dest={dest}, expected={expected}, got={result}")

def demonstrate_connectivity_check():
    """Demonstrate connectivity checking process"""
    print("\n=== Connectivity Check Demo ===")
    
    n = 6
    edges = [[0,1],[0,2],[3,5],[5,4],[4,3]]
    source = 0
    destination = 5
    
    print(f"Graph: n={n}, edges={edges}")
    print(f"Query: Path from {source} to {destination}?")
    
    # Union-Find approach
    print(f"\nUnion-Find approach:")
    uf = UnionFind(n)
    
    print(f"Initial components: {list(range(n))}")
    
    for i, (u, v) in enumerate(edges):
        print(f"\nStep {i+1}: Union({u}, {v})")
        uf.union(u, v)
        
        # Show current components
        components = defaultdict(list)
        for node in range(n):
            root = uf.find(node)
            components[root].append(node)
        
        print(f"  Components: {list(components.values())}")
        
        # Check if source and destination are connected
        if uf.connected(source, destination):
            print(f"  ✅ {source} and {destination} are now connected!")
            break
    
    final_result = uf.connected(source, destination)
    print(f"\nFinal result: {final_result}")
    
    # BFS approach for comparison
    print(f"\nBFS approach verification:")
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    print(f"Adjacency list: {dict(graph)}")
    
    queue = deque([source])
    visited = {source}
    found = False
    
    print(f"BFS from {source}:")
    step = 0
    while queue and not found:
        step += 1
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            if node == destination:
                found = True
                break
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        print(f"  Level {step}: {current_level}")
        if found:
            print(f"  🎯 Found destination {destination}!")
            break
    
    if not found:
        print(f"  ❌ Destination {destination} not reachable")

def analyze_connectivity_algorithms():
    """Analyze different connectivity algorithms"""
    print("\n=== Connectivity Algorithms Analysis ===")
    
    print("Problem Characteristics:")
    print("• Undirected graph connectivity query")
    print("• Single source-destination pair")
    print("• Binary result: connected or not")
    print("• No need for actual path")
    
    print("\nAlgorithm Comparison:")
    
    print("\n1. **Union-Find:**")
    print("   • Time: O(E α(V)) preprocessing + O(α(V)) ≈ O(1) query")
    print("   • Space: O(V)")
    print("   • Best for: Multiple connectivity queries")
    print("   • Advantage: Amortized constant query time")
    print("   • Disadvantage: Preprocessing overhead for single query")
    
    print("\n2. **DFS (Recursive):**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E) + O(V) recursion")
    print("   • Best for: Single query, sparse graphs")
    print("   • Advantage: No preprocessing, intuitive")
    print("   • Disadvantage: Recursion stack overflow risk")
    
    print("\n3. **BFS:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Best for: Single query, finds shortest path")
    print("   • Advantage: Level-by-level exploration")
    print("   • Disadvantage: Queue overhead")
    
    print("\n4. **Iterative DFS:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Best for: Single query, avoid recursion")
    print("   • Advantage: Stack-based, no recursion limit")
    print("   • Disadvantage: Slightly more complex")
    
    print("\nWhen to Use Each:")
    print("• **Union-Find:** Multiple connectivity queries")
    print("• **DFS:** Single query, need to explore deeply")
    print("• **BFS:** Single query, want shortest path")
    print("• **Iterative DFS:** Large graphs, recursion concerns")

def demonstrate_graph_components():
    """Demonstrate connected components"""
    print("\n=== Connected Components Demo ===")
    
    n = 8
    edges = [[0,1],[1,2],[3,4],[5,6]]
    
    print(f"Graph: n={n}, edges={edges}")
    
    # Find all components using Union-Find
    uf = UnionFind(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    # Group nodes by component
    components = defaultdict(list)
    for node in range(n):
        root = uf.find(node)
        components[root].append(node)
    
    component_list = list(components.values())
    print(f"Connected components: {component_list}")
    
    # Test connectivity between different pairs
    test_pairs = [(0, 2), (0, 3), (3, 4), (5, 7), (1, 6)]
    
    print(f"\nConnectivity tests:")
    for source, dest in test_pairs:
        connected = uf.connected(source, dest)
        print(f"  {source} ↔ {dest}: {connected}")

def compare_early_termination():
    """Compare early termination strategies"""
    print("\n=== Early Termination Comparison ===")
    
    print("Early Termination Strategies:")
    
    print("\n1. **Union-Find with Early Check:**")
    print("   • Check connectivity after each union")
    print("   • Stop when source-destination connected")
    print("   • Good for: Large edge sets, early connections")
    print("   • Trade-off: Extra checks vs fewer operations")
    
    print("\n2. **DFS/BFS Early Exit:**")
    print("   • Stop immediately when destination found")
    print("   • Natural early termination")
    print("   • Good for: Sparse graphs, nearby destinations")
    print("   • Trade-off: Always optimal for single query")
    
    print("\n3. **Bidirectional Search:**")
    print("   • Search from both source and destination")
    print("   • Meet in the middle")
    print("   • Good for: Long paths, balanced search")
    print("   • Trade-off: More complex implementation")
    
    print("\nPerformance Considerations:")
    print("• **Graph Density:** Dense graphs favor Union-Find")
    print("• **Query Pattern:** Multiple queries favor Union-Find")
    print("• **Path Length:** Short paths favor DFS/BFS")
    print("• **Memory Constraints:** DFS uses less space")
    print("• **Implementation Simplicity:** BFS most straightforward")

if __name__ == "__main__":
    test_valid_path()
    demonstrate_connectivity_check()
    analyze_connectivity_algorithms()
    demonstrate_graph_components()
    compare_early_termination()

"""
Union-Find Concepts:
1. Dynamic Connectivity Queries
2. Connected Components Detection
3. Path Existence Without Path Finding
4. Amortized Constant Time Operations

Key Problem Insights:
- Binary connectivity query (path exists or not)
- No need for actual path reconstruction
- Union-Find perfect for multiple queries
- DFS/BFS optimal for single queries

Algorithm Strategy:
1. Build connectivity structure (Union-Find/Graph)
2. Query connectivity between source and destination
3. Return boolean result
4. Consider early termination optimizations

Union-Find Advantages:
- O(α(V)) ≈ O(1) query time after preprocessing
- Handles dynamic edge additions efficiently
- Perfect for multiple connectivity queries
- Path compression gives excellent performance

Alternative Approaches:
- DFS: O(V+E) single query, intuitive
- BFS: O(V+E) single query, level-order
- Iterative DFS: Avoids recursion limits
- Bidirectional: Optimize for long paths

Real-world Applications:
- Network connectivity testing
- Social network friend suggestions
- Transportation route existence
- Circuit connectivity analysis
- Component membership queries

This problem demonstrates Union-Find for
basic connectivity queries in undirected graphs.
"""
