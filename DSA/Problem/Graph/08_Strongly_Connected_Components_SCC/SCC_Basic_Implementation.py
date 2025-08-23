"""
Strongly Connected Components (SCC) - Basic Implementation
Difficulty: Easy to Medium

This file provides comprehensive implementations of fundamental SCC algorithms
including Kosaraju's Algorithm, Tarjan's Algorithm, and related concepts.

Strongly Connected Component: A maximal set of vertices such that there is a path
from each vertex to every other vertex in the component.

Key Algorithms:
1. Kosaraju's Algorithm (DFS-based, uses graph transpose)
2. Tarjan's Algorithm (Single DFS with low-link values)
3. Path-based Strong Component Algorithm
4. SCC Applications and Analysis Tools

Applications:
- Dependency analysis in software systems
- Social network analysis (mutual reachability)
- Circuit analysis and design verification
- Compiler optimization (loop detection)
- Web graph analysis and PageRank computation
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class StronglyConnectedComponents:
    """Comprehensive SCC algorithms and analysis tools"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset internal state for new computation"""
        self.visited = set()
        self.finish_stack = []
        self.scc_id = {}
        self.scc_count = 0
        self.discovery_time = {}
        self.low_link = {}
        self.time = 0
        self.stack = []
        self.on_stack = set()
    
    def kosaraju_scc(self, graph: Dict[int, List[int]]) -> Tuple[int, Dict[int, int]]:
        """
        Kosaraju's Algorithm for finding SCCs
        
        Two-pass algorithm:
        1. DFS on original graph to get finish times
        2. DFS on transpose graph in reverse finish order
        
        Time: O(V + E)
        Space: O(V + E)
        """
        self.reset()
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        # Phase 1: DFS on original graph to compute finish times
        for vertex in vertices:
            if vertex not in self.visited:
                self._dfs_finish_time(graph, vertex)
        
        # Phase 2: Create transpose graph
        transpose = self._create_transpose(graph)
        
        # Phase 3: DFS on transpose in reverse finish order
        self.visited.clear()
        scc_count = 0
        
        while self.finish_stack:
            vertex = self.finish_stack.pop()
            if vertex not in self.visited:
                self._dfs_assign_scc(transpose, vertex, scc_count)
                scc_count += 1
        
        return scc_count, self.scc_id.copy()
    
    def tarjan_scc(self, graph: Dict[int, List[int]]) -> Tuple[int, Dict[int, int]]:
        """
        Tarjan's Algorithm for finding SCCs
        
        Single-pass algorithm using DFS with low-link values.
        Uses explicit stack to track current path.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset()
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        scc_count = 0
        
        for vertex in vertices:
            if vertex not in self.discovery_time:
                scc_count = self._tarjan_dfs(graph, vertex, scc_count)
        
        return scc_count, self.scc_id.copy()
    
    def path_based_scc(self, graph: Dict[int, List[int]]) -> Tuple[int, Dict[int, int]]:
        """
        Path-based Strong Component Algorithm
        
        Alternative to Tarjan's algorithm using path stack.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset()
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        path_stack = []
        scc_count = 0
        
        for vertex in vertices:
            if vertex not in self.discovery_time:
                scc_count = self._path_based_dfs(graph, vertex, path_stack, scc_count)
        
        return scc_count, self.scc_id.copy()
    
    def iterative_scc(self, graph: Dict[int, List[int]]) -> Tuple[int, Dict[int, int]]:
        """
        Iterative implementation of Tarjan's algorithm
        
        Avoids recursion using explicit stacks.
        
        Time: O(V + E)
        Space: O(V)
        """
        self.reset()
        vertices = set()
        for u in graph:
            vertices.add(u)
            for v in graph[u]:
                vertices.add(v)
        
        scc_count = 0
        
        for start_vertex in vertices:
            if start_vertex not in self.discovery_time:
                scc_count = self._iterative_tarjan_dfs(graph, start_vertex, scc_count)
        
        return scc_count, self.scc_id.copy()
    
    def _dfs_finish_time(self, graph: Dict[int, List[int]], vertex: int):
        """DFS to compute finish times for Kosaraju's algorithm"""
        self.visited.add(vertex)
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in self.visited:
                self._dfs_finish_time(graph, neighbor)
        
        self.finish_stack.append(vertex)
    
    def _create_transpose(self, graph: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Create transpose (reverse) of the graph"""
        transpose = defaultdict(list)
        
        for u in graph:
            for v in graph[u]:
                transpose[v].append(u)
        
        return transpose
    
    def _dfs_assign_scc(self, graph: Dict[int, List[int]], vertex: int, scc_id: int):
        """DFS to assign SCC IDs in Kosaraju's algorithm"""
        self.visited.add(vertex)
        self.scc_id[vertex] = scc_id
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in self.visited:
                self._dfs_assign_scc(graph, neighbor, scc_id)
    
    def _tarjan_dfs(self, graph: Dict[int, List[int]], vertex: int, scc_count: int) -> int:
        """DFS for Tarjan's algorithm"""
        # Initialize discovery time and low-link
        self.discovery_time[vertex] = self.low_link[vertex] = self.time
        self.time += 1
        
        # Push to stack and mark as on stack
        self.stack.append(vertex)
        self.on_stack.add(vertex)
        
        # Explore neighbors
        for neighbor in graph.get(vertex, []):
            if neighbor not in self.discovery_time:
                # Tree edge - recurse
                scc_count = self._tarjan_dfs(graph, neighbor, scc_count)
                self.low_link[vertex] = min(self.low_link[vertex], self.low_link[neighbor])
            elif neighbor in self.on_stack:
                # Back edge to vertex on stack
                self.low_link[vertex] = min(self.low_link[vertex], self.discovery_time[neighbor])
        
        # Check if vertex is root of SCC
        if self.low_link[vertex] == self.discovery_time[vertex]:
            # Pop SCC from stack
            while True:
                w = self.stack.pop()
                self.on_stack.remove(w)
                self.scc_id[w] = scc_count
                if w == vertex:
                    break
            scc_count += 1
        
        return scc_count
    
    def _path_based_dfs(self, graph: Dict[int, List[int]], vertex: int, path_stack: List[int], scc_count: int) -> int:
        """DFS for path-based SCC algorithm"""
        self.discovery_time[vertex] = self.time
        self.time += 1
        
        path_stack.append(vertex)
        self.stack.append(vertex)
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in self.discovery_time:
                scc_count = self._path_based_dfs(graph, neighbor, path_stack, scc_count)
            elif neighbor not in self.scc_id:
                # Remove vertices from path stack until neighbor
                while path_stack and self.discovery_time[path_stack[-1]] > self.discovery_time[neighbor]:
                    path_stack.pop()
        
        # Check if vertex is component root
        if path_stack and path_stack[-1] == vertex:
            path_stack.pop()
            # Pop SCC from stack
            while True:
                w = self.stack.pop()
                self.scc_id[w] = scc_count
                if w == vertex:
                    break
            scc_count += 1
        
        return scc_count
    
    def _iterative_tarjan_dfs(self, graph: Dict[int, List[int]], start: int, scc_count: int) -> int:
        """Iterative version of Tarjan's DFS"""
        # Stack contains (vertex, neighbor_index, is_returning)
        dfs_stack = [(start, 0, False)]
        
        while dfs_stack:
            vertex, neighbor_idx, is_returning = dfs_stack.pop()
            
            if is_returning:
                # Returning from recursive call
                if neighbor_idx > 0:
                    neighbors = list(graph.get(vertex, []))
                    if neighbor_idx <= len(neighbors):
                        neighbor = neighbors[neighbor_idx - 1]
                        self.low_link[vertex] = min(self.low_link[vertex], self.low_link[neighbor])
                
                # Continue with next neighbor
                neighbors = list(graph.get(vertex, []))
                while neighbor_idx < len(neighbors):
                    neighbor = neighbors[neighbor_idx]
                    neighbor_idx += 1
                    
                    if neighbor not in self.discovery_time:
                        # Push return marker and recursive call
                        dfs_stack.append((vertex, neighbor_idx, True))
                        dfs_stack.append((neighbor, 0, False))
                        break
                    elif neighbor in self.on_stack:
                        self.low_link[vertex] = min(self.low_link[vertex], self.discovery_time[neighbor])
                else:
                    # Finished processing all neighbors
                    if self.low_link[vertex] == self.discovery_time[vertex]:
                        # Found SCC root
                        while True:
                            w = self.stack.pop()
                            self.on_stack.remove(w)
                            self.scc_id[w] = scc_count
                            if w == vertex:
                                break
                        scc_count += 1
            else:
                # First time visiting vertex
                if vertex in self.discovery_time:
                    continue
                
                self.discovery_time[vertex] = self.low_link[vertex] = self.time
                self.time += 1
                self.stack.append(vertex)
                self.on_stack.add(vertex)
                
                # Process first neighbor or return immediately
                neighbors = list(graph.get(vertex, []))
                if neighbors and neighbor_idx < len(neighbors):
                    neighbor = neighbors[neighbor_idx]
                    neighbor_idx += 1
                    
                    if neighbor not in self.discovery_time:
                        dfs_stack.append((vertex, neighbor_idx, True))
                        dfs_stack.append((neighbor, 0, False))
                    else:
                        if neighbor in self.on_stack:
                            self.low_link[vertex] = min(self.low_link[vertex], self.discovery_time[neighbor])
                        dfs_stack.append((vertex, neighbor_idx, True))
                else:
                    # No neighbors - check if SCC root
                    if self.low_link[vertex] == self.discovery_time[vertex]:
                        w = self.stack.pop()
                        self.on_stack.remove(w)
                        self.scc_id[w] = scc_count
                        scc_count += 1
        
        return scc_count

def analyze_scc_structure(graph: Dict[int, List[int]], scc_id: Dict[int, int]) -> Dict:
    """Analyze structure of strongly connected components"""
    # Group vertices by SCC
    scc_groups = defaultdict(list)
    for vertex, component in scc_id.items():
        scc_groups[component].append(vertex)
    
    # Analyze each component
    analysis = {
        'num_components': len(scc_groups),
        'component_sizes': [len(vertices) for vertices in scc_groups.values()],
        'largest_component': max(len(vertices) for vertices in scc_groups.values()) if scc_groups else 0,
        'trivial_components': sum(1 for vertices in scc_groups.values() if len(vertices) == 1),
        'non_trivial_components': sum(1 for vertices in scc_groups.values() if len(vertices) > 1),
        'scc_groups': dict(scc_groups)
    }
    
    # Build condensation graph
    condensation = build_condensation_graph(graph, scc_id)
    analysis['condensation_graph'] = condensation
    analysis['condensation_edges'] = sum(len(neighbors) for neighbors in condensation.values())
    
    return analysis

def build_condensation_graph(graph: Dict[int, List[int]], scc_id: Dict[int, int]) -> Dict[int, Set[int]]:
    """Build condensation graph from SCCs"""
    condensation = defaultdict(set)
    
    for u in graph:
        for v in graph[u]:
            scc_u = scc_id.get(u)
            scc_v = scc_id.get(v)
            
            if scc_u is not None and scc_v is not None and scc_u != scc_v:
                condensation[scc_u].add(scc_v)
    
    return {k: v for k, v in condensation.items()}

def is_strongly_connected(graph: Dict[int, List[int]]) -> bool:
    """Check if entire graph is strongly connected"""
    scc = StronglyConnectedComponents()
    num_components, _ = scc.kosaraju_scc(graph)
    return num_components == 1

def find_strongly_connected_pairs(graph: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    """Find all pairs of vertices that are strongly connected"""
    scc = StronglyConnectedComponents()
    _, scc_id = scc.kosaraju_scc(graph)
    
    # Group vertices by component
    components = defaultdict(list)
    for vertex, component in scc_id.items():
        components[component].append(vertex)
    
    # Generate all pairs within each component
    pairs = []
    for vertices in components.values():
        if len(vertices) > 1:
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    pairs.append((vertices[i], vertices[j]))
    
    return pairs

def test_scc_algorithms():
    """Test all SCC algorithms with various graphs"""
    print("=== Testing SCC Algorithms ===")
    
    # Test cases
    test_graphs = [
        # Simple cycle
        {0: [1], 1: [2], 2: [0]},
        
        # Two components
        {0: [1], 1: [2], 2: [0], 3: [4], 4: [3]},
        
        # Complex graph
        {0: [1], 1: [2, 3], 2: [0], 3: [4], 4: [5, 7], 5: [6], 6: [4, 7], 7: []},
        
        # Single vertex
        {0: []},
        
        # Disconnected components
        {0: [1], 1: [0], 2: [3], 3: [2], 4: []},
    ]
    
    scc = StronglyConnectedComponents()
    algorithms = [
        ("Kosaraju's", scc.kosaraju_scc),
        ("Tarjan's", scc.tarjan_scc),
        ("Path-based", scc.path_based_scc),
        ("Iterative", scc.iterative_scc),
    ]
    
    for i, graph in enumerate(test_graphs):
        print(f"\nTest Graph {i+1}: {graph}")
        
        results = []
        for name, algorithm in algorithms:
            try:
                scc.reset()
                num_components, scc_assignments = algorithm(graph)
                results.append((name, num_components, scc_assignments))
            except Exception as e:
                results.append((name, f"Error: {e}", {}))
        
        # Display results
        for name, num_components, assignments in results:
            print(f"  {name}: {num_components} components")
            if isinstance(assignments, dict) and assignments:
                print(f"    Assignments: {assignments}")
        
        # Verify all algorithms give same result
        if len(set(r[1] for r in results if isinstance(r[1], int))) == 1:
            print("  ✓ All algorithms agree")
        else:
            print("  ✗ Algorithms disagree!")

def demonstrate_scc_applications():
    """Demonstrate practical applications of SCC"""
    print("\n=== SCC Applications Demo ===")
    
    print("\n1. **Dependency Analysis:**")
    # Software module dependencies
    modules = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': ['A'],  # Circular dependency!
        'E': ['F'],
        'F': ['E']   # Another cycle
    }
    
    # Convert to numeric for algorithm
    module_map = {name: i for i, name in enumerate(modules.keys())}
    reverse_map = {i: name for name, i in module_map.items()}
    
    numeric_graph = {}
    for module, deps in modules.items():
        numeric_graph[module_map[module]] = [module_map[dep] for dep in deps]
    
    scc = StronglyConnectedComponents()
    num_components, scc_id = scc.kosaraju_scc(numeric_graph)
    
    print(f"  Module dependencies: {modules}")
    print(f"  Found {num_components} strongly connected components:")
    
    # Group by component
    components = defaultdict(list)
    for vertex, component in scc_id.items():
        components[component].append(reverse_map[vertex])
    
    for comp_id, module_list in components.items():
        if len(module_list) > 1:
            print(f"    Circular dependency: {module_list}")
        else:
            print(f"    Independent module: {module_list}")
    
    print("\n2. **Social Network Analysis:**")
    # Mutual following relationships
    social_graph = {
        0: [1, 2],     # User 0 follows 1, 2
        1: [2],        # User 1 follows 2
        2: [0],        # User 2 follows 0 (creates cycle)
        3: [4],        # User 3 follows 4
        4: [3, 5],     # User 4 follows 3, 5
        5: []          # User 5 follows nobody
    }
    
    num_components, scc_id = scc.kosaraju_scc(social_graph)
    analysis = analyze_scc_structure(social_graph, scc_id)
    
    print(f"  Social network: {social_graph}")
    print(f"  Mutual following groups: {analysis['scc_groups']}")
    print(f"  Number of mutual groups: {num_components}")

def demonstrate_condensation_graph():
    """Demonstrate condensation graph construction"""
    print("\n=== Condensation Graph Demo ===")
    
    graph = {
        0: [1], 1: [2], 2: [0],  # First SCC: {0, 1, 2}
        3: [4], 4: [3],          # Second SCC: {3, 4}
        2: [3],                  # Edge between SCCs
        5: [6], 6: [5],          # Third SCC: {5, 6}
        4: [5]                   # Another edge between SCCs
    }
    
    # Fix graph representation
    full_graph = {
        0: [1],
        1: [2],
        2: [0, 3],
        3: [4],
        4: [3, 5],
        5: [6],
        6: [5]
    }
    
    scc = StronglyConnectedComponents()
    num_components, scc_id = scc.kosaraju_scc(full_graph)
    
    print(f"Original graph: {full_graph}")
    print(f"SCC assignments: {scc_id}")
    
    condensation = build_condensation_graph(full_graph, scc_id)
    print(f"Condensation graph: {dict(condensation)}")
    
    print(f"\nProperties:")
    print(f"  Original vertices: {len(set(scc_id.keys()))}")
    print(f"  Condensed vertices: {num_components}")
    print(f"  Condensed edges: {sum(len(neighbors) for neighbors in condensation.values())}")
    print(f"  Is DAG: {is_dag(condensation)}")

def is_dag(graph: Dict[int, Set[int]]) -> bool:
    """Check if graph is a DAG using topological sort"""
    in_degree = defaultdict(int)
    vertices = set()
    
    for u in graph:
        vertices.add(u)
        for v in graph[u]:
            vertices.add(v)
            in_degree[v] += 1
    
    queue = deque([v for v in vertices if in_degree[v] == 0])
    processed = 0
    
    while queue:
        u = queue.popleft()
        processed += 1
        
        for v in graph.get(u, set()):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    return processed == len(vertices)

def analyze_algorithm_performance():
    """Analyze performance characteristics of SCC algorithms"""
    print("\n=== Algorithm Performance Analysis ===")
    
    print("SCC Algorithm Comparison:")
    
    print("\n1. **Kosaraju's Algorithm:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E) - needs transpose graph")
    print("   • Two DFS passes")
    print("   • Simple to understand and implement")
    print("   • Good for educational purposes")
    
    print("\n2. **Tarjan's Algorithm:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V) - no transpose needed")
    print("   • Single DFS pass")
    print("   • More complex but space-efficient")
    print("   • Preferred for large graphs")
    
    print("\n3. **Path-based Algorithm:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V)")
    print("   • Alternative to Tarjan's")
    print("   • Uses path stack instead of low-link")
    print("   • Good theoretical properties")
    
    print("\n4. **Iterative Versions:**")
    print("   • Same complexity as recursive")
    print("   • Avoid stack overflow for deep graphs")
    print("   • More complex implementation")
    print("   • Better for production systems")
    
    print("\nAlgorithm Selection:")
    print("• **Kosaraju's:** Educational, simple graphs")
    print("• **Tarjan's:** Production, large graphs, memory-constrained")
    print("• **Path-based:** Alternative when Tarjan's is complex")
    print("• **Iterative:** Deep graphs, stack overflow concerns")

if __name__ == "__main__":
    test_scc_algorithms()
    demonstrate_scc_applications()
    demonstrate_condensation_graph()
    analyze_algorithm_performance()

"""
Strongly Connected Components and Graph Analysis Concepts:
1. SCC Identification with Kosaraju's and Tarjan's Algorithms
2. Condensation Graph Construction and DAG Properties
3. Dependency Analysis and Circular Reference Detection
4. Social Network Analysis and Mutual Connectivity
5. Algorithm Performance and Implementation Trade-offs

Key Problem Insights:
- SCCs represent maximal mutually reachable vertex sets
- Condensation graph is always a DAG
- Multiple algorithms with same complexity but different trade-offs
- Essential for dependency analysis and graph structure understanding

Algorithm Strategy:
1. Kosaraju's: Two DFS passes with graph transpose
2. Tarjan's: Single DFS with low-link values and stack
3. Path-based: Alternative stack-based approach
4. All achieve O(V + E) time complexity

SCC Applications:
- Software dependency analysis and circular reference detection
- Social network analysis for mutual following/friendship groups
- Circuit analysis and loop detection
- Web graph analysis and community detection
- Compiler optimization and dead code elimination

Real-world Applications:
- Software engineering and build systems
- Social media and network analysis
- Circuit design and verification
- Web search and PageRank algorithms
- Distributed systems and consensus protocols

This implementation provides complete SCC algorithm mastery
with practical applications in software engineering and network analysis.
"""
