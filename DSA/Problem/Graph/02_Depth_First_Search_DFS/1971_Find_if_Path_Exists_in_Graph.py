"""
1971. Find if Path Exists in Graph
Difficulty: Easy

Problem:
There is a bi-directional graph with n vertices, where each vertex is labeled from 0 to n - 1 (inclusive). 
The edges in the graph are represented as a 2D integer array edges, where each edges[i] = [ui, vi] 
denotes a bi-directional edge between vertex ui and vertex vi. Every vertex pair is connected by at 
most one edge, and no vertex has an edge to itself.

You want to determine if there is a valid path that exists from vertex source to vertex destination.

Given edges and the integers n, source, and destination, return true if there is a valid path from 
source to destination, or false otherwise.

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
from collections import defaultdict, deque

class Solution:
    def validPath_approach1_dfs_recursive(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 1: DFS with Recursion
        
        Build adjacency list and use DFS to check if destination is reachable from source.
        
        Time: O(V + E) where V = n, E = len(edges)
        Space: O(V + E) for adjacency list + recursion stack
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
            if node == destination:
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            
            # Explore all neighbors
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            
            return False
        
        return dfs(source)
    
    def validPath_approach2_dfs_iterative(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 2: DFS with Iteration (Stack-based)
        
        Use explicit stack to avoid recursion depth issues.
        
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
        stack = [source]
        
        while stack:
            node = stack.pop()
            
            if node == destination:
                return True
            
            if node in visited:
                continue
            
            visited.add(node)
            
            # Add unvisited neighbors to stack
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return False
    
    def validPath_approach3_bfs(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 3: BFS (Level-order traversal)
        
        Use BFS to explore graph level by level.
        
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
        queue = deque([source])
        visited.add(source)
        
        while queue:
            node = queue.popleft()
            
            # Check all neighbors
            for neighbor in graph[node]:
                if neighbor == destination:
                    return True
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return False
    
    def validPath_approach4_union_find(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 4: Union-Find (Disjoint Set Union)
        
        Build connected components and check if source and destination are in same component.
        
        Time: O(E * α(V)) where α is inverse Ackermann function
        Space: O(V)
        """
        if source == destination:
            return True
        
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
            
            def connected(self, x, y):
                return self.find(x) == self.find(y)
        
        uf = UnionFind(n)
        
        # Union all connected vertices
        for u, v in edges:
            uf.union(u, v)
        
        return uf.connected(source, destination)
    
    def validPath_approach5_bidirectional_search(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 5: Bidirectional Search
        
        Search from both source and destination simultaneously until they meet.
        Can be faster for large graphs.
        
        Time: O(V + E) - often better in practice
        Space: O(V + E)
        """
        if source == destination:
            return True
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Two sets for visited nodes from each direction
        visited_from_source = {source}
        visited_from_dest = {destination}
        
        # Two queues for BFS from each direction
        queue_source = deque([source])
        queue_dest = deque([destination])
        
        while queue_source or queue_dest:
            # Expand from source side
            if queue_source:
                node = queue_source.popleft()
                for neighbor in graph[node]:
                    if neighbor in visited_from_dest:
                        return True  # Paths meet
                    
                    if neighbor not in visited_from_source:
                        visited_from_source.add(neighbor)
                        queue_source.append(neighbor)
            
            # Expand from destination side
            if queue_dest:
                node = queue_dest.popleft()
                for neighbor in graph[node]:
                    if neighbor in visited_from_source:
                        return True  # Paths meet
                    
                    if neighbor not in visited_from_dest:
                        visited_from_dest.add(neighbor)
                        queue_dest.append(neighbor)
        
        return False

def test_valid_path():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, source, destination, expected)
        (3, [[0,1],[1,2],[2,0]], 0, 2, True),
        (6, [[0,1],[0,2],[3,5],[5,4],[4,3]], 0, 5, False),
        (1, [], 0, 0, True),  # Single node
        (2, [[0,1]], 0, 1, True),  # Direct connection
        (2, [], 0, 1, False),  # No connection
        (4, [[0,1],[2,3]], 0, 3, False),  # Two separate components
        (5, [[0,1],[1,2],[2,3],[3,4]], 0, 4, True),  # Path chain
        (3, [[0,1],[0,2]], 1, 2, True),  # Star graph
    ]
    
    approaches = [
        ("DFS Recursive", solution.validPath_approach1_dfs_recursive),
        ("DFS Iterative", solution.validPath_approach2_dfs_iterative),
        ("BFS", solution.validPath_approach3_bfs),
        ("Union-Find", solution.validPath_approach4_union_find),
        ("Bidirectional Search", solution.validPath_approach5_bidirectional_search),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, source, dest, expected) in enumerate(test_cases):
            result = func(n, edges, source, dest)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, edges={edges}, {source}->{dest}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_path_finding():
    """Demonstrate path finding process"""
    print("\n=== Path Finding Demonstration ===")
    
    n = 6
    edges = [[0,1],[1,2],[3,4],[4,5]]
    source, destination = 0, 5
    
    print(f"Graph: n={n}, edges={edges}")
    print(f"Finding path from {source} to {destination}")
    
    # Build and visualize graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    print(f"\nAdjacency List:")
    for node in range(n):
        neighbors = graph[node] if node in graph else []
        print(f"  Node {node}: {neighbors}")
    
    # Trace DFS path finding
    visited = set()
    path = []
    
    def dfs_trace(node, target):
        if node in visited:
            return False
        
        visited.add(node)
        path.append(node)
        
        if node == target:
            return True
        
        for neighbor in graph[node]:
            if dfs_trace(neighbor, target):
                return True
        
        path.pop()  # Backtrack
        return False
    
    found = dfs_trace(source, destination)
    
    print(f"\nDFS Exploration:")
    print(f"  Visited nodes: {sorted(visited)}")
    print(f"  Path found: {found}")
    if found:
        print(f"  Path: {' -> '.join(map(str, path))}")
    
    # Analyze connected components
    all_visited = set()
    components = []
    
    def dfs_component(node, component):
        if node in all_visited:
            return
        
        all_visited.add(node)
        component.append(node)
        
        for neighbor in graph[node]:
            dfs_component(neighbor, component)
    
    for node in range(n):
        if node not in all_visited:
            component = []
            dfs_component(node, component)
            if component:
                components.append(sorted(component))
    
    print(f"\nConnected Components: {components}")
    print(f"Same component: {any(source in comp and destination in comp for comp in components)}")

def visualize_graph_connectivity():
    """Create visual representation of graph connectivity"""
    print("\n=== Graph Connectivity Visualization ===")
    
    examples = [
        ("Connected", 4, [[0,1],[1,2],[2,3]], 0, 3),
        ("Disconnected", 4, [[0,1],[2,3]], 0, 3),
        ("Star", 4, [[0,1],[0,2],[0,3]], 1, 3),
        ("Cycle", 4, [[0,1],[1,2],[2,3],[3,0]], 0, 2),
    ]
    
    for name, n, edges, source, dest in examples:
        print(f"\n{name} Graph:")
        print(f"  Edges: {edges}")
        print(f"  Path {source} -> {dest}: ", end="")
        
        # Build graph
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # Check connectivity using BFS
        if source == dest:
            print("True (same node)")
            continue
        
        visited = set()
        queue = deque([source])
        visited.add(source)
        found = False
        
        while queue and not found:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor == dest:
                    found = True
                    break
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        print("True" if found else "False")
        
        # Show connected components
        all_visited = set()
        components = []
        
        for node in range(n):
            if node not in all_visited:
                component = []
                stack = [node]
                
                while stack:
                    curr = stack.pop()
                    if curr not in all_visited:
                        all_visited.add(curr)
                        component.append(curr)
                        for neighbor in graph[curr]:
                            if neighbor not in all_visited:
                                stack.append(neighbor)
                
                if component:
                    components.append(sorted(component))
        
        print(f"  Components: {components}")

def analyze_algorithm_performance():
    """Analyze performance of different pathfinding algorithms"""
    print("\n=== Algorithm Performance Analysis ===")
    
    algorithms = [
        ("DFS Recursive", "O(V+E)", "O(V+E)", "Simple, natural", "Stack overflow risk"),
        ("DFS Iterative", "O(V+E)", "O(V+E)", "No recursion limit", "Explicit stack management"),
        ("BFS", "O(V+E)", "O(V+E)", "Level-order, shortest path", "Queue overhead"),
        ("Union-Find", "O(E*α(V))", "O(V)", "Good for multiple queries", "Overkill for single query"),
        ("Bidirectional", "O(V+E)", "O(V+E)", "Can be faster in practice", "More complex implementation"),
    ]
    
    print(f"{'Algorithm':<15} {'Time':<10} {'Space':<10} {'Advantages':<25} {'Disadvantages'}")
    print("-" * 85)
    
    for alg, time_comp, space_comp, advantages, disadvantages in algorithms:
        print(f"{alg:<15} {time_comp:<10} {space_comp:<10} {advantages:<25} {disadvantages}")
    
    print(f"\nRecommendations:")
    print(f"- DFS Recursive: Most natural for simple path finding")
    print(f"- BFS: When you need shortest path (unweighted graphs)")
    print(f"- Union-Find: Multiple connectivity queries on same graph")
    print(f"- Bidirectional: Large graphs with known source and destination")

def real_world_applications():
    """Demonstrate real-world applications of path finding"""
    print("\n=== Real-World Applications ===")
    
    applications = [
        ("Social Networks", "Find connection between users", "Friend recommendations"),
        ("Computer Networks", "Check if nodes can communicate", "Network troubleshooting"),
        ("Transportation", "Route existence between cities", "GPS navigation systems"),
        ("Game Development", "Check if player can reach target", "AI pathfinding"),
        ("Web Crawling", "Check if page is reachable", "SEO analysis"),
        ("Circuit Design", "Verify electrical connectivity", "PCB validation"),
        ("Dependency Analysis", "Check if modules are connected", "Build systems"),
        ("Biological Networks", "Protein interaction pathways", "Drug discovery"),
    ]
    
    print(f"{'Domain':<20} {'Problem':<35} {'Application'}")
    print("-" * 75)
    
    for domain, problem, application in applications:
        print(f"{domain:<20} {problem:<35} {application}")

if __name__ == "__main__":
    test_valid_path()
    demonstrate_path_finding()
    visualize_graph_connectivity()
    analyze_algorithm_performance()
    real_world_applications()

"""
Graph Theory Concepts:
1. Graph Connectivity and Reachability
2. Path Existence vs Path Finding
3. Connected Components
4. Graph Traversal Algorithms

Key Path Finding Concepts:
- Reachability: Can we get from A to B?
- Graph connectivity: Which nodes can reach which other nodes?
- Connected components: Groups of mutually reachable nodes
- Traversal strategies: DFS, BFS, bidirectional search

Algorithm Comparison:
- DFS: Memory efficient, simple implementation
- BFS: Finds shortest path (unweighted), level-order
- Union-Find: Excellent for multiple connectivity queries
- Bidirectional: Can reduce search space significantly

Optimization Techniques:
- Early termination when target is found
- Bidirectional search to reduce search space
- Union-Find for preprocessing connectivity
- Visited set to avoid cycles

Real-world Applications:
- Network connectivity testing
- Social network analysis
- Route planning and navigation
- Dependency resolution
- Reachability analysis in distributed systems

This is a fundamental graph problem that forms the basis for more
complex pathfinding and connectivity algorithms.
"""
