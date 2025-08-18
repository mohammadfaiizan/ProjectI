"""
Graph Validation Problems Collection
Difficulty: Medium

This file contains various graph validation and analysis problems that demonstrate
fundamental graph theory concepts and validation techniques.

Problems included:
1. Valid Tree Detection
2. Graph Isomorphism Check (Simple)
3. Bipartite Graph Validation
4. Connected Graph Validation
5. Acyclic Graph Validation
6. Complete Graph Validation
7. Planar Graph Detection (Basic)
8. Regular Graph Validation
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class GraphValidator:
    """
    Collection of graph validation and analysis methods
    """
    
    def is_valid_tree(self, n: int, edges: List[List[int]]) -> bool:
        """
        Problem: Valid Tree Detection
        
        A valid tree must satisfy:
        1. Exactly n-1 edges
        2. Connected graph
        3. No cycles
        
        Time: O(N)
        Space: O(N)
        """
        # Quick check: tree with n nodes has exactly n-1 edges
        if len(edges) != n - 1:
            return False
        
        if n == 1:
            return True
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # Check connectivity using DFS
        visited = set()
        
        def dfs(node):
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(0)
        
        # Valid tree if all nodes are visited (connected) and edge count is correct
        return len(visited) == n
    
    def is_bipartite(self, graph: List[List[int]]) -> bool:
        """
        Problem: Bipartite Graph Validation
        
        A graph is bipartite if vertices can be colored with 2 colors
        such that no adjacent vertices have the same color.
        
        Time: O(V + E)
        Space: O(V)
        """
        n = len(graph)
        color = [-1] * n  # -1: uncolored, 0/1: two colors
        
        for start in range(n):
            if color[start] == -1:
                # BFS coloring
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    node = queue.popleft()
                    
                    for neighbor in graph[node]:
                        if color[neighbor] == -1:
                            color[neighbor] = 1 - color[node]
                            queue.append(neighbor)
                        elif color[neighbor] == color[node]:
                            return False  # Same color for adjacent nodes
        
        return True
    
    def is_connected(self, n: int, edges: List[List[int]]) -> bool:
        """
        Problem: Connected Graph Validation
        
        Check if all vertices are reachable from any starting vertex.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if n <= 1:
            return True
        
        # Build adjacency list
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        # DFS from node 0
        visited = set()
        stack = [0]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return len(visited) == n
    
    def is_acyclic(self, n: int, edges: List[List[int]], directed: bool = False) -> bool:
        """
        Problem: Acyclic Graph Validation
        
        Check if graph contains no cycles.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if directed:
            return self._is_acyclic_directed(n, edges)
        else:
            return self._is_acyclic_undirected(n, edges)
    
    def _is_acyclic_directed(self, n: int, edges: List[List[int]]) -> bool:
        """Check if directed graph is acyclic (DAG)"""
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
        
        # DFS with three colors: white (0), gray (1), black (2)
        color = [0] * n
        
        def dfs(node):
            if color[node] == 1:  # Gray: back edge found (cycle)
                return False
            if color[node] == 2:  # Black: already processed
                return True
            
            color[node] = 1  # Gray: currently processing
            
            for neighbor in adj[node]:
                if not dfs(neighbor):
                    return False
            
            color[node] = 2  # Black: finished processing
            return True
        
        for i in range(n):
            if color[i] == 0:
                if not dfs(i):
                    return False
        
        return True
    
    def _is_acyclic_undirected(self, n: int, edges: List[List[int]]) -> bool:
        """Check if undirected graph is acyclic"""
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        
        visited = set()
        
        def dfs(node, parent):
            visited.add(node)
            
            for neighbor in adj[node]:
                if neighbor == parent:  # Skip edge to parent
                    continue
                
                if neighbor in visited:  # Back edge found (cycle)
                    return False
                
                if not dfs(neighbor, node):
                    return False
            
            return True
        
        for i in range(n):
            if i not in visited:
                if not dfs(i, -1):
                    return False
        
        return True
    
    def is_complete_graph(self, n: int, edges: List[List[int]]) -> bool:
        """
        Problem: Complete Graph Validation
        
        A complete graph has edges between every pair of vertices.
        Complete graph with n vertices has exactly n*(n-1)/2 edges.
        
        Time: O(E)
        Space: O(E)
        """
        expected_edges = n * (n - 1) // 2
        
        if len(edges) != expected_edges:
            return False
        
        # Check if all possible edges exist
        edge_set = set()
        for a, b in edges:
            edge = tuple(sorted([a, b]))
            if edge in edge_set:  # Duplicate edge
                return False
            edge_set.add(edge)
        
        # Verify all pairs exist
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    return False
        
        return True
    
    def is_regular_graph(self, n: int, edges: List[List[int]], k: Optional[int] = None) -> Tuple[bool, int]:
        """
        Problem: Regular Graph Validation
        
        A k-regular graph has all vertices with degree k.
        If k is not specified, check if graph is regular and return the degree.
        
        Time: O(E)
        Space: O(V)
        
        Returns: (is_regular, degree)
        """
        degree = [0] * n
        
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        
        if not degree:  # No edges
            return True, 0
        
        expected_degree = degree[0] if k is None else k
        
        for d in degree:
            if d != expected_degree:
                return False, -1
        
        return True, expected_degree
    
    def simple_graph_isomorphism_check(self, graph1: List[List[int]], graph2: List[List[int]]) -> bool:
        """
        Problem: Simple Graph Isomorphism Check
        
        Basic isomorphism check using degree sequence.
        This is not a complete solution but catches many non-isomorphic cases.
        
        Time: O(V + E)
        Space: O(V)
        """
        n1, n2 = len(graph1), len(graph2)
        
        if n1 != n2:
            return False
        
        # Calculate degree sequences
        deg1 = [len(neighbors) for neighbors in graph1]
        deg2 = [len(neighbors) for neighbors in graph2]
        
        # Sort degree sequences
        deg1.sort()
        deg2.sort()
        
        return deg1 == deg2
    
    def has_euler_path(self, n: int, edges: List[List[int]]) -> bool:
        """
        Problem: Euler Path Existence
        
        Euler path exists if:
        - Graph is connected (ignoring isolated vertices)
        - At most 2 vertices have odd degree
        
        Time: O(V + E)
        Space: O(V + E)
        """
        if not edges:
            return n <= 1
        
        # Build graph and calculate degrees
        adj = defaultdict(list)
        degree = [0] * n
        vertices_with_edges = set()
        
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
            degree[a] += 1
            degree[b] += 1
            vertices_with_edges.add(a)
            vertices_with_edges.add(b)
        
        # Check connectivity of vertices with edges
        if not self._is_connected_subset(vertices_with_edges, adj):
            return False
        
        # Count vertices with odd degree
        odd_degree_count = sum(1 for d in degree if d % 2 == 1)
        
        return odd_degree_count <= 2
    
    def _is_connected_subset(self, vertices: Set[int], adj: Dict[int, List[int]]) -> bool:
        """Check if subset of vertices is connected"""
        if not vertices:
            return True
        
        start = next(iter(vertices))
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adj[node]:
                    if neighbor in vertices and neighbor not in visited:
                        stack.append(neighbor)
        
        return len(visited) == len(vertices)

def test_graph_validations():
    """Test all validation methods"""
    validator = GraphValidator()
    
    print("=== Graph Validation Tests ===")
    
    # Test cases for different validations
    test_cases = [
        # Valid tree
        ("Valid Tree", 4, [[0,1],[1,2],[2,3]], "is_valid_tree", True),
        ("Invalid Tree (cycle)", 4, [[0,1],[1,2],[2,3],[3,0]], "is_valid_tree", False),
        ("Invalid Tree (too many edges)", 3, [[0,1],[1,2],[0,2],[2,0]], "is_valid_tree", False),
        
        # Bipartite
        ("Bipartite", [[1,3],[0,2],[1,3],[0,2]], "is_bipartite", True),
        ("Not Bipartite", [[1,2,3],[0,2],[0,1,3],[0,2]], "is_bipartite", False),
        
        # Connected
        ("Connected", 4, [[0,1],[1,2],[2,3]], "is_connected", True),
        ("Disconnected", 4, [[0,1],[2,3]], "is_connected", False),
        
        # Complete graph
        ("Complete K4", 4, [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], "is_complete_graph", True),
        ("Incomplete", 4, [[0,1],[0,2],[1,2]], "is_complete_graph", False),
    ]
    
    for name, *args in test_cases:
        if args[-1] == "is_bipartite":
            result = validator.is_bipartite(args[0])
            expected = args[1]
        elif args[-1] == "is_valid_tree":
            result = validator.is_valid_tree(args[0], args[1])
            expected = args[2]
        elif args[-1] == "is_connected":
            result = validator.is_connected(args[0], args[1])
            expected = args[2]
        elif args[-1] == "is_complete_graph":
            result = validator.is_complete_graph(args[0], args[1])
            expected = args[2]
        
        status = "✓" if result == expected else "✗"
        print(f"{status} {name}: Expected {expected}, Got {result}")

def demonstrate_graph_properties():
    """Demonstrate analysis of various graph properties"""
    print("\n=== Graph Properties Analysis ===")
    
    validator = GraphValidator()
    
    graphs = [
        ("Path Graph P4", 4, [[0,1],[1,2],[2,3]]),
        ("Cycle Graph C4", 4, [[0,1],[1,2],[2,3],[3,0]]),
        ("Complete Graph K4", 4, [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]),
        ("Star Graph S4", 4, [[0,1],[0,2],[0,3]]),
        ("Disconnected", 5, [[0,1],[2,3]]),
    ]
    
    for name, n, edges in graphs:
        print(f"\n{name}:")
        print(f"  Vertices: {n}, Edges: {len(edges)}")
        
        # Analyze properties
        is_tree = validator.is_valid_tree(n, edges)
        is_connected = validator.is_connected(n, edges)
        is_acyclic = validator.is_acyclic(n, edges, directed=False)
        is_complete = validator.is_complete_graph(n, edges)
        is_regular, degree = validator.is_regular_graph(n, edges)
        has_euler = validator.has_euler_path(n, edges)
        
        print(f"  Properties:")
        print(f"    Tree: {is_tree}")
        print(f"    Connected: {is_connected}")
        print(f"    Acyclic: {is_acyclic}")
        print(f"    Complete: {is_complete}")
        print(f"    Regular: {is_regular} (degree: {degree})")
        print(f"    Has Euler path: {has_euler}")

def analyze_graph_families():
    """Analyze properties of well-known graph families"""
    print("\n=== Graph Family Analysis ===")
    
    validator = GraphValidator()
    
    # Generate different graph families
    def generate_path(n):
        """Generate path graph P_n"""
        return [[i, i+1] for i in range(n-1)]
    
    def generate_cycle(n):
        """Generate cycle graph C_n"""
        edges = [[i, i+1] for i in range(n-1)]
        edges.append([n-1, 0])
        return edges
    
    def generate_complete(n):
        """Generate complete graph K_n"""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append([i, j])
        return edges
    
    def generate_star(n):
        """Generate star graph S_n (center + n leaves)"""
        return [[0, i] for i in range(1, n)]
    
    families = [
        ("Path P5", 5, generate_path(5)),
        ("Cycle C5", 5, generate_cycle(5)),
        ("Complete K5", 5, generate_complete(5)),
        ("Star S5", 5, generate_star(5)),
    ]
    
    print(f"{'Graph':<12} {'Tree':<6} {'Conn.':<6} {'Acyc.':<6} {'Comp.':<6} {'Reg.':<6} {'Euler':<6}")
    print("-" * 54)
    
    for name, n, edges in families:
        props = {
            'tree': validator.is_valid_tree(n, edges),
            'connected': validator.is_connected(n, edges),
            'acyclic': validator.is_acyclic(n, edges, directed=False),
            'complete': validator.is_complete_graph(n, edges),
            'regular': validator.is_regular_graph(n, edges)[0],
            'euler': validator.has_euler_path(n, edges),
        }
        
        print(f"{name:<12} {str(props['tree']):<6} {str(props['connected']):<6} " +
              f"{str(props['acyclic']):<6} {str(props['complete']):<6} " +
              f"{str(props['regular']):<6} {str(props['euler']):<6}")

if __name__ == "__main__":
    test_graph_validations()
    demonstrate_graph_properties()
    analyze_graph_families()

"""
Graph Theory Concepts Covered:
1. Tree Properties and Validation
2. Graph Connectivity Analysis
3. Bipartite Graph Detection
4. Cycle Detection (Directed/Undirected)
5. Complete Graph Properties
6. Regular Graph Analysis
7. Euler Path/Circuit Conditions
8. Basic Graph Isomorphism

Key Algorithms:
- DFS/BFS for connectivity
- Graph coloring for bipartiteness
- Union-Find for connectivity
- Cycle detection algorithms
- Degree sequence analysis

Real-world Applications:
- Network topology validation
- Data structure verification
- Algorithm correctness checking
- Graph database integrity
- Social network analysis
- Transportation network validation
"""
