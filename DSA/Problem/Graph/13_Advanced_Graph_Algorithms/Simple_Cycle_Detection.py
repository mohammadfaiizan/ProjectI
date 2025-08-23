"""
Simple Cycle Detection - Multiple Approaches
Difficulty: Easy

Cycle detection in graphs is fundamental for many algorithms.
This file implements various cycle detection methods for both
directed and undirected graphs.

Key Concepts:
1. DFS-based Cycle Detection
2. Union-Find Cycle Detection
3. Topological Sort for DAG
4. Color-based Detection
5. Path Tracking Methods
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque

class SimpleCycleDetection:
    """Comprehensive cycle detection algorithms"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.is_directed = False
    
    def add_edge(self, u: int, v: int, directed: bool = False):
        """Add edge to graph"""
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)
        
        if not directed:
            self.graph[v].append(u)
        
        self.is_directed = directed
    
    def has_cycle_dfs_undirected(self) -> bool:
        """
        Approach 1: DFS Cycle Detection for Undirected Graph
        
        Use DFS with parent tracking to detect back edges.
        
        Time: O(V + E), Space: O(V)
        """
        visited = set()
        
        def dfs(node: int, parent: int) -> bool:
            visited.add(node)
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    # Back edge found (cycle)
                    return True
            
            return False
        
        # Check all components
        for vertex in self.vertices:
            if vertex not in visited:
                if dfs(vertex, -1):
                    return True
        
        return False
    
    def has_cycle_dfs_directed(self) -> bool:
        """
        Approach 2: DFS Cycle Detection for Directed Graph
        
        Use DFS with recursion stack to detect back edges.
        
        Time: O(V + E), Space: O(V)
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        
        def dfs(node: int) -> bool:
            color[node] = GRAY
            
            for neighbor in self.graph[node]:
                if color[neighbor] == GRAY:
                    # Back edge found (cycle)
                    return True
                elif color[neighbor] == WHITE and dfs(neighbor):
                    return True
            
            color[node] = BLACK
            return False
        
        # Check all components
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                if dfs(vertex):
                    return True
        
        return False
    
    def has_cycle_union_find(self) -> bool:
        """
        Approach 3: Union-Find Cycle Detection (Undirected Only)
        
        Use Union-Find to detect cycles by checking if endpoints
        of an edge are already connected.
        
        Time: O(E * α(V)), Space: O(V)
        """
        if self.is_directed:
            return self.has_cycle_dfs_directed()
        
        parent = {v: v for v in self.vertices}
        rank = {v: 0 for v in self.vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            
            if px == py:
                return False  # Cycle detected
            
            if rank[px] < rank[py]:
                px, py = py, px
            
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            
            return True
        
        # Process edges
        processed_edges = set()
        
        for u in self.graph:
            for v in self.graph[u]:
                if (u, v) not in processed_edges and (v, u) not in processed_edges:
                    processed_edges.add((u, v))
                    
                    if not union(u, v):
                        return True
        
        return False
    
    def has_cycle_topological_sort(self) -> bool:
        """
        Approach 4: Topological Sort for Directed Graphs
        
        If topological sort is possible, no cycle exists.
        
        Time: O(V + E), Space: O(V)
        """
        if not self.is_directed:
            return self.has_cycle_dfs_undirected()
        
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for u in self.graph:
            for v in self.graph[u]:
                in_degree[v] += 1
        
        # Initialize queue with vertices having in-degree 0
        queue = deque([v for v in self.vertices if in_degree[v] == 0])
        processed = 0
        
        while queue:
            node = queue.popleft()
            processed += 1
            
            for neighbor in self.graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If all vertices processed, no cycle
        return processed != len(self.vertices)
    
    def find_cycle_path_dfs(self) -> Optional[List[int]]:
        """
        Approach 5: Find Actual Cycle Path using DFS
        
        Returns the vertices in a cycle if one exists.
        
        Time: O(V + E), Space: O(V)
        """
        if self.is_directed:
            return self._find_cycle_path_directed()
        else:
            return self._find_cycle_path_undirected()
    
    def _find_cycle_path_undirected(self) -> Optional[List[int]]:
        """Find cycle path in undirected graph"""
        visited = set()
        parent = {}
        
        def dfs(node: int, par: int) -> Optional[List[int]]:
            visited.add(node)
            parent[node] = par
            
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    cycle = dfs(neighbor, node)
                    if cycle:
                        return cycle
                elif neighbor != par:
                    # Found cycle, reconstruct path
                    cycle = [neighbor, node]
                    current = par
                    
                    while current != neighbor and current != -1:
                        cycle.append(current)
                        current = parent[current]
                    
                    return cycle
            
            return None
        
        for vertex in self.vertices:
            if vertex not in visited:
                cycle = dfs(vertex, -1)
                if cycle:
                    return cycle
        
        return None
    
    def _find_cycle_path_directed(self) -> Optional[List[int]]:
        """Find cycle path in directed graph"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        parent = {}
        
        def dfs(node: int) -> Optional[List[int]]:
            color[node] = GRAY
            
            for neighbor in self.graph[node]:
                if color[neighbor] == GRAY:
                    # Found back edge, reconstruct cycle
                    cycle = [neighbor]
                    current = node
                    
                    while current != neighbor:
                        cycle.append(current)
                        current = parent[current]
                    
                    return cycle
                elif color[neighbor] == WHITE:
                    parent[neighbor] = node
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
            
            color[node] = BLACK
            return None
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                cycle = dfs(vertex)
                if cycle:
                    return cycle
        
        return None
    
    def count_cycles_simple(self) -> int:
        """
        Approach 6: Count Simple Cycles (Approximation)
        
        Count cycles using DFS (may count some cycles multiple times).
        
        Time: O(V + E), Space: O(V)
        """
        if self.is_directed:
            return self._count_cycles_directed()
        else:
            return self._count_cycles_undirected()
    
    def _count_cycles_undirected(self) -> int:
        """Count cycles in undirected graph (approximation)"""
        visited = set()
        cycle_count = 0
        
        def dfs(node: int, parent: int, path: Set[int]) -> int:
            visited.add(node)
            path.add(node)
            count = 0
            
            for neighbor in self.graph[node]:
                if neighbor in path and neighbor != parent:
                    count += 1
                elif neighbor not in visited:
                    count += dfs(neighbor, node, path)
            
            path.remove(node)
            return count
        
        for vertex in self.vertices:
            if vertex not in visited:
                cycle_count += dfs(vertex, -1, set())
        
        return cycle_count // 2  # Each cycle counted twice
    
    def _count_cycles_directed(self) -> int:
        """Count cycles in directed graph (approximation)"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        cycle_count = 0
        
        def dfs(node: int, path: Set[int]) -> int:
            color[node] = GRAY
            path.add(node)
            count = 0
            
            for neighbor in self.graph[node]:
                if neighbor in path:
                    count += 1
                elif color[neighbor] == WHITE:
                    count += dfs(neighbor, path)
            
            path.remove(node)
            color[node] = BLACK
            return count
        
        for vertex in self.vertices:
            if color[vertex] == WHITE:
                cycle_count += dfs(vertex, set())
        
        return cycle_count
    
    def get_cycle_analysis(self) -> Dict:
        """
        Approach 7: Comprehensive Cycle Analysis
        
        Return detailed information about cycles in the graph.
        
        Time: O(V + E), Space: O(V)
        """
        analysis = {
            'has_cycle': False,
            'cycle_path': None,
            'is_directed': self.is_directed,
            'vertices': len(self.vertices),
            'edges': sum(len(adj) for adj in self.graph.values()) // (1 if self.is_directed else 2),
            'is_connected': self._is_connected(),
            'components': len(self._get_connected_components()),
        }
        
        # Detect cycle
        if self.is_directed:
            analysis['has_cycle'] = self.has_cycle_dfs_directed()
        else:
            analysis['has_cycle'] = self.has_cycle_dfs_undirected()
        
        # Find cycle path if exists
        if analysis['has_cycle']:
            analysis['cycle_path'] = self.find_cycle_path_dfs()
            analysis['cycle_length'] = len(analysis['cycle_path']) if analysis['cycle_path'] else 0
        
        # Additional properties
        if not self.is_directed:
            analysis['is_tree'] = (analysis['edges'] == len(self.vertices) - 1 and 
                                 analysis['is_connected'] and not analysis['has_cycle'])
            analysis['is_forest'] = not analysis['has_cycle']
        else:
            analysis['is_dag'] = not analysis['has_cycle']
        
        return analysis
    
    def _is_connected(self) -> bool:
        """Check if graph is connected (undirected) or weakly connected (directed)"""
        if not self.vertices:
            return True
        
        # For directed graphs, check weak connectivity
        if self.is_directed:
            # Create undirected version
            undirected_graph = defaultdict(set)
            for u in self.graph:
                for v in self.graph[u]:
                    undirected_graph[u].add(v)
                    undirected_graph[v].add(u)
            
            # BFS on undirected version
            start = next(iter(self.vertices))
            visited = set()
            queue = deque([start])
            visited.add(start)
            
            while queue:
                node = queue.popleft()
                for neighbor in undirected_graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == len(self.vertices)
        else:
            # Standard connectivity check for undirected
            start = next(iter(self.vertices))
            visited = set()
            queue = deque([start])
            visited.add(start)
            
            while queue:
                node = queue.popleft()
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return len(visited) == len(self.vertices)
    
    def _get_connected_components(self) -> List[Set[int]]:
        """Get connected components"""
        visited = set()
        components = []
        
        for vertex in self.vertices:
            if vertex not in visited:
                component = set()
                queue = deque([vertex])
                
                while queue:
                    node = queue.popleft()
                    if node not in visited:
                        visited.add(node)
                        component.add(node)
                        
                        for neighbor in self.graph[node]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(component)
        
        return components

def test_cycle_detection():
    """Test cycle detection algorithms"""
    print("=== Testing Cycle Detection ===")
    
    # Test cases: (edges, directed, expected_cycle, description)
    test_cases = [
        # Undirected graphs
        ([(0, 1), (1, 2), (2, 0)], False, True, "Undirected triangle"),
        ([(0, 1), (1, 2), (2, 3)], False, False, "Undirected path"),
        ([(0, 1), (1, 2), (2, 3), (3, 1)], False, True, "Undirected cycle"),
        
        # Directed graphs
        ([(0, 1), (1, 2), (2, 0)], True, True, "Directed triangle"),
        ([(0, 1), (1, 2), (2, 3)], True, False, "Directed path"),
        ([(0, 1), (1, 2), (2, 3), (3, 1)], True, True, "Directed cycle"),
        ([(0, 1), (1, 2), (0, 2)], True, False, "Directed DAG"),
        
        # Edge cases
        ([], False, False, "Empty graph"),
        ([(0, 1)], False, False, "Single edge"),
    ]
    
    for edges, directed, expected_cycle, description in test_cases:
        print(f"\n--- {description} ---")
        
        graph = SimpleCycleDetection()
        for u, v in edges:
            graph.add_edge(u, v, directed)
        
        print(f"Edges: {edges}, Directed: {directed}")
        
        # Test different algorithms
        algorithms = [
            ("DFS", graph.has_cycle_dfs_directed if directed else graph.has_cycle_dfs_undirected),
            ("Union-Find", graph.has_cycle_union_find),
            ("Topological", graph.has_cycle_topological_sort),
        ]
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func()
                status = "✓" if result == expected_cycle else "✗"
                print(f"{alg_name:12} | {status} | Has Cycle: {result}")
            except Exception as e:
                print(f"{alg_name:12} | ERROR: {str(e)[:30]}")
        
        # Show cycle path if exists
        cycle_path = graph.find_cycle_path_dfs()
        if cycle_path:
            print(f"Cycle Path: {cycle_path}")
        
        # Show analysis
        analysis = graph.get_cycle_analysis()
        print(f"Analysis: {analysis['has_cycle']} cycle, {analysis['vertices']} vertices, "
              f"{analysis['edges']} edges, {analysis['components']} components")

def demonstrate_cycle_concepts():
    """Demonstrate cycle detection concepts"""
    print("\n=== Cycle Detection Concepts ===")
    
    print("Key Concepts:")
    print("1. **Undirected Graphs:** Cycle = back edge to non-parent")
    print("2. **Directed Graphs:** Cycle = back edge in DFS tree")
    print("3. **Union-Find:** Cycle when endpoints already connected")
    print("4. **Topological Sort:** Impossible if cycle exists")
    
    print("\nAlgorithm Properties:")
    print("• DFS: O(V + E) time, detects any cycle")
    print("• Union-Find: O(E * α(V)) time, good for undirected")
    print("• Topological: O(V + E) time, directed graphs only")
    print("• BFS: Can be adapted for cycle detection")
    
    print("\nApplications:")
    print("• Deadlock detection in systems")
    print("• Dependency resolution")
    print("• Circuit analysis")
    print("• Scheduling problems")
    print("• Graph validation")

def analyze_cycle_complexity():
    """Analyze complexity of cycle detection algorithms"""
    print("\n=== Cycle Detection Complexity ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **DFS-Based:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V) - recursion stack")
    print("   • Pros: Simple, works for both directed/undirected")
    print("   • Cons: Recursion depth limited")
    
    print("\n2. **Union-Find:**")
    print("   • Time: O(E * α(V)) - nearly linear")
    print("   • Space: O(V)")
    print("   • Pros: Good for undirected, incremental")
    print("   • Cons: Undirected graphs only")
    
    print("\n3. **Topological Sort:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V)")
    print("   • Pros: Natural for directed graphs")
    print("   • Cons: Directed graphs only")
    
    print("\n4. **Iterative DFS:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V) - explicit stack")
    print("   • Pros: No recursion limits")
    print("   • Cons: More complex implementation")
    
    print("\nRecommendations:")
    print("• Undirected: DFS or Union-Find")
    print("• Directed: DFS or Topological Sort")
    print("• Large graphs: Iterative approaches")
    print("• Incremental: Union-Find")

if __name__ == "__main__":
    test_cycle_detection()
    demonstrate_cycle_concepts()
    analyze_cycle_complexity()

"""
Simple Cycle Detection - Key Insights:

1. **Fundamental Concepts:**
   - Cycle = closed path with no repeated vertices (except start/end)
   - Different detection methods for directed vs undirected graphs
   - Back edges indicate cycles in DFS traversal
   - Union-Find detects cycles by connectivity

2. **Algorithm Categories:**
   - DFS-based: Use recursion stack or explicit stack
   - Union-Find: Incremental connectivity checking
   - Topological: Impossible ordering indicates cycle
   - BFS-based: Level-order cycle detection

3. **Key Techniques:**
   - Parent tracking in undirected graphs
   - Color coding (white/gray/black) in directed graphs
   - Path reconstruction for cycle identification
   - Component analysis for disconnected graphs

4. **Complexity Considerations:**
   - Linear time O(V + E) for most algorithms
   - Space varies: O(V) for recursion/stack
   - Union-Find: Nearly linear with path compression
   - Choice depends on graph type and requirements

5. **Applications:**
   - Deadlock detection in operating systems
   - Dependency cycle detection in build systems
   - Circuit analysis and validation
   - Scheduling and resource allocation
   - Graph structure validation

Cycle detection is fundamental to many graph algorithms
and provides the basis for more complex graph analysis.
"""
