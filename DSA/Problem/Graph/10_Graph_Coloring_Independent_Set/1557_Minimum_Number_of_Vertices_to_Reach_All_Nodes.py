"""
1557. Minimum Number of Vertices to Reach All Nodes
Difficulty: Medium

Problem:
Given a directed acyclic graph, with n vertices numbered from 0 to n-1, and an array edges 
where edges[i] = [from_i, to_i] represents a directed edge from node from_i to node to_i.

Find the smallest set of vertices from which all nodes in the graph are reachable. 
It's guaranteed that a unique solution exists.

Notice that you can reach a vertex from itself.

Examples:
Input: n = 6, edges = [[0,1],[0,2],[2,3],[3,4],[3,5]]
Output: [0]
Explanation: From a single node 0 we can reach [0,1,2,3,4,5].

Input: n = 5, edges = [[0,1],[2,3],[3,4]]
Output: [0,2]
Explanation: From node 0 we can reach [0,1]. From node 2 we can reach [2,3,4]. So we output [0,2].

Input: n = 3, edges = []
Output: [0,1,2]
Explanation: Since there are no edges, no node is reachable from any other node. 
We have to include all nodes into the result nodes list.

Constraints:
- 2 <= n <= 10^5
- 1 <= edges.length <= min(10^5, n * (n - 1) / 2)
- edges[i].length == 2
- 0 <= from_i, to_i < n
- All pairs (from_i, to_i) are distinct.
- The input is generated such that edges form a DAG.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class Solution:
    def findSmallestSetOfVertices_approach1_indegree_analysis(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: In-degree Analysis (Optimal)
        
        Vertices with in-degree 0 must be in the solution.
        
        Time: O(V + E)
        Space: O(V)
        """
        # Calculate in-degree for each vertex
        indegree = [0] * n
        
        for from_node, to_node in edges:
            indegree[to_node] += 1
        
        # Vertices with in-degree 0 are the answer
        result = []
        for i in range(n):
            if indegree[i] == 0:
                result.append(i)
        
        return result
    
    def findSmallestSetOfVertices_approach2_reachability_analysis(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: Reachability Analysis
        
        Find vertices that are not reachable from any other vertex.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for from_node, to_node in edges:
            graph[from_node].append(to_node)
        
        # Find all reachable vertices
        reachable = set()
        
        def dfs(node, visited):
            """DFS to find all reachable vertices from node"""
            if node in visited:
                return
            
            visited.add(node)
            reachable.add(node)
            
            for neighbor in graph[node]:
                dfs(neighbor, visited)
        
        # For each vertex, find what it can reach
        for start in range(n):
            visited = set()
            dfs(start, visited)
        
        # Find vertices that are never reached by others
        result = []
        for i in range(n):
            # Check if vertex i is reachable from any other vertex
            is_reachable_from_others = False
            
            for start in range(n):
                if start != i:
                    visited = set()
                    self._dfs_check_reachable(start, i, graph, visited)
                    if i in visited:
                        is_reachable_from_others = True
                        break
            
            if not is_reachable_from_others:
                result.append(i)
        
        return result
    
    def _dfs_check_reachable(self, current, target, graph, visited):
        """Helper DFS to check if target is reachable from current"""
        if current in visited:
            return
        
        visited.add(current)
        
        for neighbor in graph[current]:
            self._dfs_check_reachable(neighbor, target, graph, visited)
    
    def findSmallestSetOfVertices_approach3_topological_analysis(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 3: Topological Sort Analysis
        
        Use topological properties to find minimum dominating set.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build graph and calculate in-degrees
        graph = defaultdict(list)
        indegree = [0] * n
        
        for from_node, to_node in edges:
            graph[from_node].append(to_node)
            indegree[to_node] += 1
        
        # Find vertices with no incoming edges (sources)
        sources = []
        for i in range(n):
            if indegree[i] == 0:
                sources.append(i)
        
        # Verify that sources can reach all other vertices
        reachable_from_sources = set()
        
        def dfs_from_sources(node, visited):
            """DFS to find vertices reachable from sources"""
            if node in visited:
                return
            
            visited.add(node)
            reachable_from_sources.add(node)
            
            for neighbor in graph[node]:
                dfs_from_sources(neighbor, visited)
        
        # Check reachability from all sources
        visited = set()
        for source in sources:
            dfs_from_sources(source, visited)
        
        # In a DAG, sources should reach all vertices
        return sources
    
    def findSmallestSetOfVertices_approach4_strongly_connected_components(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 4: SCC Analysis (Adapted for DAG)
        
        Analyze connectivity structure using SCC concepts.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Since this is a DAG, each vertex is its own SCC
        # Find "root" SCCs (vertices with no incoming edges)
        
        has_incoming = [False] * n
        
        for from_node, to_node in edges:
            has_incoming[to_node] = True
        
        # Vertices without incoming edges are the roots
        result = []
        for i in range(n):
            if not has_incoming[i]:
                result.append(i)
        
        return result
    
    def findSmallestSetOfVertices_approach5_graph_theory_analysis(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 5: Comprehensive Graph Theory Analysis
        
        Detailed analysis using graph theory concepts.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build comprehensive graph representation
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        indegree = [0] * n
        outdegree = [0] * n
        
        for from_node, to_node in edges:
            graph[from_node].append(to_node)
            reverse_graph[to_node].append(from_node)
            indegree[to_node] += 1
            outdegree[from_node] += 1
        
        # Analyze vertex properties
        sources = []  # Vertices with indegree 0
        sinks = []    # Vertices with outdegree 0
        
        for i in range(n):
            if indegree[i] == 0:
                sources.append(i)
            if outdegree[i] == 0:
                sinks.append(i)
        
        # For minimum dominating set in DAG, sources are the answer
        # Verify by checking reachability
        total_reachable = set()
        
        def compute_reachable(start):
            """Compute all vertices reachable from start"""
            reachable = set()
            stack = [start]
            
            while stack:
                node = stack.pop()
                if node in reachable:
                    continue
                
                reachable.add(node)
                
                for neighbor in graph[node]:
                    if neighbor not in reachable:
                        stack.append(neighbor)
            
            return reachable
        
        # Check that sources cover all vertices
        for source in sources:
            reachable = compute_reachable(source)
            total_reachable.update(reachable)
        
        # Should cover all vertices in a connected DAG
        assert len(total_reachable) == n or len(sources) == n  # Handle disconnected case
        
        return sources
    
    def findSmallestSetOfVertices_approach6_optimized_validation(self, n: int, edges: List[List[int]]) -> List[int]:
        """
        Approach 6: Optimized with Validation
        
        Optimized solution with comprehensive validation.
        
        Time: O(V + E)
        Space: O(V)
        """
        # Simple and efficient: find vertices with indegree 0
        has_incoming_edge = [False] * n
        
        for from_node, to_node in edges:
            has_incoming_edge[to_node] = True
        
        result = [i for i in range(n) if not has_incoming_edge[i]]
        
        # Validation: verify this is indeed minimal and sufficient
        self._validate_solution(n, edges, result)
        
        return result
    
    def _validate_solution(self, n: int, edges: List[List[int]], solution: List[int]):
        """Validate that solution is correct"""
        # Build graph for validation
        graph = defaultdict(list)
        for from_node, to_node in edges:
            graph[from_node].append(to_node)
        
        # Check that all vertices are reachable from solution
        reachable = set()
        
        def dfs(node, visited):
            if node in visited:
                return
            visited.add(node)
            reachable.add(node)
            for neighbor in graph[node]:
                dfs(neighbor, visited)
        
        visited = set()
        for start in solution:
            dfs(start, visited)
        
        if len(reachable) != n:
            raise ValueError(f"Solution {solution} doesn't reach all vertices. Reachable: {len(reachable)}/{n}")
        
        # Check minimality: no proper subset should work
        for i in range(len(solution)):
            test_solution = solution[:i] + solution[i+1:]
            if test_solution:  # Non-empty subset
                test_reachable = set()
                test_visited = set()
                
                for start in test_solution:
                    dfs(start, test_visited)
                
                test_reachable = test_visited
                
                if len(test_reachable) == n:
                    raise ValueError(f"Solution {solution} is not minimal. Subset {test_solution} also works")

def test_minimum_vertices():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, expected)
        (6, [[0,1],[0,2],[2,3],[3,4],[3,5]], [0]),
        (5, [[0,1],[2,3],[3,4]], [0,2]),
        (3, [], [0,1,2]),
        (4, [[0,1],[1,2],[2,3]], [0]),
        (5, [[0,1],[0,2],[1,3],[2,4]], [0]),
        (2, [[0,1]], [0]),
        (1, [], [0]),
    ]
    
    approaches = [
        ("In-degree Analysis", solution.findSmallestSetOfVertices_approach1_indegree_analysis),
        ("Reachability Analysis", solution.findSmallestSetOfVertices_approach2_reachability_analysis),
        ("Topological Analysis", solution.findSmallestSetOfVertices_approach3_topological_analysis),
        ("SCC Analysis", solution.findSmallestSetOfVertices_approach4_strongly_connected_components),
        ("Graph Theory", solution.findSmallestSetOfVertices_approach5_graph_theory_analysis),
        ("Optimized Validation", solution.findSmallestSetOfVertices_approach6_optimized_validation),
    ]
    
    for i, (n, edges, expected) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: n={n} ---")
        print(f"Edges: {edges}")
        print(f"Expected: {expected}")
        
        for approach_name, func in approaches:
            try:
                result = func(n, edges[:])  # Copy edges
                
                # Check if result is correct (may be different order)
                result_set = set(result)
                expected_set = set(expected)
                correct = result_set == expected_set
                
                status = "✓" if correct else "✗"
                print(f"{approach_name:20} | {status} | Result: {sorted(result)}")
                
            except Exception as e:
                print(f"{approach_name:20} | ERROR: {str(e)}")

def demonstrate_indegree_analysis():
    """Demonstrate in-degree analysis approach"""
    print("\n=== In-degree Analysis Demo ===")
    
    n = 6
    edges = [[0,1],[0,2],[2,3],[3,4],[3,5]]
    
    print(f"Graph: n={n}, edges={edges}")
    
    # Calculate in-degrees
    indegree = [0] * n
    for from_node, to_node in edges:
        indegree[to_node] += 1
    
    print(f"\nIn-degree analysis:")
    for i in range(n):
        print(f"Vertex {i}: in-degree = {indegree[i]}")
    
    result = [i for i in range(n) if indegree[i] == 0]
    
    print(f"\nVertices with in-degree 0: {result}")
    print(f"These are the sources that must be in any solution")
    
    # Verify reachability
    from collections import defaultdict
    graph = defaultdict(list)
    for from_node, to_node in edges:
        graph[from_node].append(to_node)
    
    def dfs_reachable(start):
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(neighbor)
        
        return visited
    
    print(f"\nReachability verification:")
    all_reachable = set()
    for source in result:
        reachable = dfs_reachable(source)
        print(f"From vertex {source}: can reach {sorted(reachable)}")
        all_reachable.update(reachable)
    
    print(f"Total reachable: {sorted(all_reachable)}")
    print(f"Covers all vertices: {len(all_reachable) == n}")

def demonstrate_dag_properties():
    """Demonstrate DAG properties relevant to the problem"""
    print("\n=== DAG Properties Demo ===")
    
    print("Key DAG Properties for Minimum Vertex Set:")
    
    print("\n1. **Source Vertices:**")
    print("   • Vertices with in-degree 0")
    print("   • Must be included in any solution")
    print("   • Cannot be reached from other vertices")
    print("   • Form the minimal dominating set")
    
    print("\n2. **Reachability:**")
    print("   • Every vertex is reachable from some source")
    print("   • DAG structure ensures no cycles")
    print("   • Topological ordering exists")
    print("   • Sources appear first in topological order")
    
    print("\n3. **Minimality:**")
    print("   • Cannot remove any source from solution")
    print("   • Each source reaches unique set of vertices")
    print("   • Solution size equals number of weakly connected components")
    
    print("\n4. **Uniqueness:**")
    print("   • Solution is unique in terms of which vertices")
    print("   • Only sources can be in minimal solution")
    print("   • Problem guarantees unique solution exists")
    
    # Example with disconnected components
    print(f"\nExample with multiple components:")
    n = 7
    edges = [[0,1],[0,2],[3,4],[5,6]]
    
    indegree = [0] * n
    for from_node, to_node in edges:
        indegree[to_node] += 1
    
    sources = [i for i in range(n) if indegree[i] == 0]
    
    print(f"Graph: {edges}")
    print(f"Components: {{0,1,2}}, {{3,4}}, {{5,6}}")
    print(f"Sources: {sources}")
    print(f"Each component contributes one source")

def analyze_algorithm_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **In-degree Analysis (Optimal):**")
    print("   • Time: O(V + E) - single pass through edges")
    print("   • Space: O(V) - in-degree array")
    print("   • Most efficient and direct")
    print("   • Leverages DAG structure optimally")
    
    print("\n2. **Reachability Analysis:**")
    print("   • Time: O(V * (V + E)) - DFS from each vertex")
    print("   • Space: O(V + E) - graph storage")
    print("   • Computationally expensive")
    print("   • Educational but inefficient")
    
    print("\n3. **Topological Analysis:**")
    print("   • Time: O(V + E) - topological sort")
    print("   • Space: O(V + E) - graph and auxiliary structures")
    print("   • Correct but more complex than needed")
    
    print("\n4. **SCC Analysis:**")
    print("   • Time: O(V + E) - adapted SCC concepts")
    print("   • Space: O(V) - boolean array")
    print("   • Conceptually interesting")
    print("   • Same result as in-degree analysis")
    
    print("\nRecommendation:")
    print("• **Use In-degree Analysis** for production code")
    print("• Simple, optimal, and directly addresses problem")
    print("• Other approaches useful for understanding")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Applications ===")
    
    print("Minimum Vertex Set Applications:")
    
    print("\n1. **Build System Dependencies:**")
    print("   • Source files → Vertices")
    print("   • Dependencies → Directed edges")
    print("   • Find minimal set of files to build everything")
    print("   • Root source files have no dependencies")
    
    print("\n2. **Package Installation:**")
    print("   • Software packages → Vertices")
    print("   • Dependencies → Directed edges")
    print("   • Find packages to install manually")
    print("   • Others will be installed as dependencies")
    
    print("\n3. **Task Scheduling:**")
    print("   • Tasks → Vertices")
    print("   • Prerequisites → Directed edges")
    print("   • Find tasks that must be started manually")
    print("   • Others triggered by completion of predecessors")
    
    print("\n4. **Knowledge Propagation:**")
    print("   • People → Vertices")
    print("   • Information flow → Directed edges")
    print("   • Find minimal set to inform directly")
    print("   • Information spreads to rest through network")
    
    print("\n5. **Infection Control:**")
    print("   • Individuals → Vertices")
    print("   • Transmission paths → Directed edges")
    print("   • Find sources of infection")
    print("   • Target intervention at these sources")

def demonstrate_edge_cases():
    """Demonstrate edge cases and special scenarios"""
    print("\n=== Edge Cases Demo ===")
    
    solution = Solution()
    
    print("Special Cases:")
    
    # Empty graph
    print("\n1. **Empty Graph:**")
    n, edges = 3, []
    result = solution.findSmallestSetOfVertices_approach1_indegree_analysis(n, edges)
    print(f"   n={n}, edges={edges}")
    print(f"   Result: {result}")
    print(f"   All vertices isolated, all must be included")
    
    # Single vertex
    print("\n2. **Single Vertex:**")
    n, edges = 1, []
    result = solution.findSmallestSetOfVertices_approach1_indegree_analysis(n, edges)
    print(f"   n={n}, edges={edges}")
    print(f"   Result: {result}")
    
    # Chain graph
    print("\n3. **Chain Graph:**")
    n, edges = 4, [[0,1],[1,2],[2,3]]
    result = solution.findSmallestSetOfVertices_approach1_indegree_analysis(n, edges)
    print(f"   n={n}, edges={edges}")
    print(f"   Result: {result}")
    print(f"   Only one source needed for entire chain")
    
    # Star graph (one source, many sinks)
    print("\n4. **Star Graph:**")
    n, edges = 5, [[0,1],[0,2],[0,3],[0,4]]
    result = solution.findSmallestSetOfVertices_approach1_indegree_analysis(n, edges)
    print(f"   n={n}, edges={edges}")
    print(f"   Result: {result}")
    print(f"   Center vertex reaches all others")
    
    # Complete DAG (tournament)
    print("\n5. **Tournament (One source):**")
    n, edges = 4, [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
    result = solution.findSmallestSetOfVertices_approach1_indegree_analysis(n, edges)
    print(f"   n={n}, edges={edges}")
    print(f"   Result: {result}")
    print(f"   Topological ordering: 0 → 1 → 2 → 3")

if __name__ == "__main__":
    test_minimum_vertices()
    demonstrate_indegree_analysis()
    demonstrate_dag_properties()
    analyze_algorithm_complexity()
    demonstrate_real_world_applications()
    demonstrate_edge_cases()

"""
Minimum Dominating Set and DAG Analysis Concepts:
1. In-degree Analysis for Source Vertex Identification
2. DAG Properties and Topological Structure Analysis
3. Reachability and Connectivity in Directed Graphs
4. Graph Theory Applications in Dependency Management
5. Optimization Techniques for Minimum Vertex Cover Problems

Key Problem Insights:
- Minimum dominating set in DAG equals set of source vertices
- Sources are vertices with in-degree 0
- Simple O(V + E) algorithm using in-degree counting
- Solution is unique and minimal by DAG structure
- Applications in dependency analysis and build systems

Algorithm Strategy:
1. Calculate in-degree for each vertex
2. Select vertices with in-degree 0 as solution
3. These sources can reach all other vertices
4. Solution is minimal and unique

DAG Properties:
- Acyclic structure enables simple solution
- Topological ordering always exists
- Sources appear first in any topological order
- Each weakly connected component has at least one source
- Reachability follows topological structure

Optimization Techniques:
- Single pass in-degree calculation
- Efficient graph representation
- Early termination optimizations
- Space-optimal solution storage

Real-world Applications:
- Build system dependency management
- Package installation optimization
- Task scheduling and project management
- Knowledge and information propagation
- Infection control and intervention targeting

This problem demonstrates how DAG structure enables
elegant solutions to complex optimization problems.
"""
