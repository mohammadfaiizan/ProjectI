"""
Hamiltonian Path Problem
Difficulty: Hard
Category: Bitmask DP - Graph Path Problems

PROBLEM DESCRIPTION:
===================
Given a directed graph, find if there exists a Hamiltonian path - a path that visits each vertex exactly once.

Unlike the Traveling Salesman Problem, a Hamiltonian path doesn't need to return to the starting vertex.

The graph is represented as an adjacency matrix where graph[i][j] = 1 if there's an edge from vertex i to vertex j.

Example 1:
Input: graph = [[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]]
Output: True
Explanation: Path 0->2->1->3 visits all vertices exactly once.

Example 2:
Input: graph = [[0,1,0],[1,0,1],[0,1,0]]
Output: True
Explanation: Path 0->1->2 visits all vertices exactly once.

Example 3:
Input: graph = [[0,0,1],[1,0,0],[0,1,0]]
Output: False
Explanation: No path can visit all vertices exactly once.

Constraints:
- 1 <= n <= 20
- graph[i][j] is 0 or 1
- graph[i][i] = 0 (no self-loops)
"""


def has_hamiltonian_path_brute_force(graph):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible permutations of vertices.
    
    Time Complexity: O(n!) - factorial explosion
    Space Complexity: O(n) - recursion stack
    """
    from itertools import permutations
    
    n = len(graph)
    if n <= 1:
        return True
    
    # Try all permutations as potential paths
    for perm in permutations(range(n)):
        valid_path = True
        
        # Check if this permutation forms a valid path
        for i in range(len(perm) - 1):
            if graph[perm[i]][perm[i + 1]] == 0:
                valid_path = False
                break
        
        if valid_path:
            return True
    
    return False


def has_hamiltonian_path_bitmask_dp(graph):
    """
    BITMASK DP APPROACH:
    ===================
    Use bitmask DP to find Hamiltonian path efficiently.
    
    Time Complexity: O(n^2 * 2^n) - optimal for this problem
    Space Complexity: O(n * 2^n) - DP table
    """
    n = len(graph)
    if n <= 1:
        return True
    
    # dp[mask][last] = True if we can visit all vertices in mask ending at vertex 'last'
    dp = [[False] * n for _ in range(1 << n)]
    
    # Base case: start from each vertex
    for i in range(n):
        dp[1 << i][i] = True
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or not dp[mask][last]:
                continue
            
            # Try extending to each unvisited vertex
            for next_vertex in range(n):
                if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                    continue
                
                new_mask = mask | (1 << next_vertex)
                dp[new_mask][next_vertex] = True
    
    # Check if any ending vertex can reach all vertices
    full_mask = (1 << n) - 1
    return any(dp[full_mask][i] for i in range(n))


def find_hamiltonian_path_with_reconstruction(graph):
    """
    HAMILTONIAN PATH WITH PATH RECONSTRUCTION:
    =========================================
    Find Hamiltonian path and return the actual path.
    
    Time Complexity: O(n^2 * 2^n) - DP computation + path reconstruction
    Space Complexity: O(n * 2^n) - DP table + parent tracking
    """
    n = len(graph)
    if n <= 1:
        return True, list(range(n))
    
    # DP with parent tracking
    dp = [[False] * n for _ in range(1 << n)]
    parent = [[(-1, -1)] * n for _ in range(1 << n)]
    
    # Initialize starting positions
    for i in range(n):
        dp[1 << i][i] = True
    
    # Fill DP table with parent tracking
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or not dp[mask][last]:
                continue
            
            for next_vertex in range(n):
                if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                    continue
                
                new_mask = mask | (1 << next_vertex)
                if not dp[new_mask][next_vertex]:
                    dp[new_mask][next_vertex] = True
                    parent[new_mask][next_vertex] = (mask, last)
    
    # Find a valid ending and reconstruct path
    full_mask = (1 << n) - 1
    
    for end_vertex in range(n):
        if dp[full_mask][end_vertex]:
            # Reconstruct path
            path = []
            mask, last = full_mask, end_vertex
            
            while mask != 0:
                path.append(last)
                if parent[mask][last] == (-1, -1):
                    break
                prev_mask, prev_last = parent[mask][last]
                mask, last = prev_mask, prev_last
            
            return True, path[::-1]
    
    return False, []


def count_hamiltonian_paths(graph):
    """
    COUNT HAMILTONIAN PATHS:
    =======================
    Count the total number of distinct Hamiltonian paths.
    
    Time Complexity: O(n^2 * 2^n) - DP computation
    Space Complexity: O(n * 2^n) - DP table
    """
    n = len(graph)
    if n <= 1:
        return 1
    
    # dp[mask][last] = number of paths visiting vertices in mask ending at 'last'
    dp = [[0] * n for _ in range(1 << n)]
    
    # Base case: start from each vertex
    for i in range(n):
        dp[1 << i][i] = 1
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == 0:
                continue
            
            for next_vertex in range(n):
                if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                    continue
                
                new_mask = mask | (1 << next_vertex)
                dp[new_mask][next_vertex] += dp[mask][last]
    
    # Sum all complete paths
    full_mask = (1 << n) - 1
    return sum(dp[full_mask][i] for i in range(n))


def hamiltonian_path_analysis(graph):
    """
    COMPREHENSIVE HAMILTONIAN PATH ANALYSIS:
    =======================================
    Analyze the graph and provide detailed insights about Hamiltonian paths.
    
    Time Complexity: O(n^2 * 2^n) - complete analysis
    Space Complexity: O(n * 2^n) - DP tables + analysis data
    """
    n = len(graph)
    
    analysis = {
        'num_vertices': n,
        'num_edges': sum(sum(row) for row in graph),
        'has_hamiltonian_path': False,
        'path_count': 0,
        'example_path': [],
        'degree_analysis': {},
        'connectivity_info': {},
        'bottlenecks': [],
        'impossible_reasons': []
    }
    
    # Basic graph analysis
    in_degrees = [sum(graph[i][j] for i in range(n)) for j in range(n)]
    out_degrees = [sum(graph[i][j] for j in range(n)) for i in range(n)]
    
    analysis['degree_analysis'] = {
        'in_degrees': in_degrees,
        'out_degrees': out_degrees,
        'isolated_vertices': [i for i in range(n) if in_degrees[i] == 0 and out_degrees[i] == 0],
        'sources': [i for i in range(n) if in_degrees[i] == 0 and out_degrees[i] > 0],
        'sinks': [i for i in range(n) if in_degrees[i] > 0 and out_degrees[i] == 0]
    }
    
    # Check for obvious impossibilities
    if len(analysis['degree_analysis']['isolated_vertices']) > 0:
        analysis['impossible_reasons'].append("Isolated vertices exist")
    
    if len(analysis['degree_analysis']['sources']) > 1:
        analysis['impossible_reasons'].append("Multiple source vertices")
    
    if len(analysis['degree_analysis']['sinks']) > 1:
        analysis['impossible_reasons'].append("Multiple sink vertices")
    
    # Run DP analysis
    dp = [[0] * n for _ in range(1 << n)]
    parent = [[(-1, -1)] * n for _ in range(1 << n)]
    
    for i in range(n):
        dp[1 << i][i] = 1
    
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == 0:
                continue
            
            for next_vertex in range(n):
                if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                    continue
                
                new_mask = mask | (1 << next_vertex)
                if dp[new_mask][next_vertex] == 0:
                    parent[new_mask][next_vertex] = (mask, last)
                dp[new_mask][next_vertex] += dp[mask][last]
    
    # Analyze results
    full_mask = (1 << n) - 1
    total_paths = sum(dp[full_mask][i] for i in range(n))
    
    analysis['path_count'] = total_paths
    analysis['has_hamiltonian_path'] = total_paths > 0
    
    # Find example path if exists
    if total_paths > 0:
        for end_vertex in range(n):
            if dp[full_mask][end_vertex] > 0:
                path = []
                mask, last = full_mask, end_vertex
                
                while mask != 0:
                    path.append(last)
                    if parent[mask][last] == (-1, -1):
                        break
                    prev_mask, prev_last = parent[mask][last]
                    mask, last = prev_mask, prev_last
                
                analysis['example_path'] = path[::-1]
                break
    
    # Identify bottlenecks (vertices that appear in all paths)
    if total_paths > 0:
        vertex_frequency = [0] * n
        
        # Count how often each vertex appears as intermediate in paths
        for mask in range(1, (1 << n) - 1):
            num_vertices = bin(mask).count('1')
            if num_vertices < 2:
                continue
            
            for vertex in range(n):
                if (mask & (1 << vertex)) and any(dp[mask][vertex] > 0 for vertex in range(n) if mask & (1 << vertex)):
                    vertex_frequency[vertex] += 1
        
        total_intermediate_positions = sum(vertex_frequency)
        if total_intermediate_positions > 0:
            for i, freq in enumerate(vertex_frequency):
                if freq > total_intermediate_positions * 0.8:  # Appears in >80% of positions
                    analysis['bottlenecks'].append(i)
    
    return analysis


def hamiltonian_path_variants():
    """
    HAMILTONIAN PATH VARIANTS:
    =========================
    Different scenarios and modifications.
    """
    
    def longest_simple_path(graph):
        """Find the longest simple path (no repeated vertices)"""
        n = len(graph)
        max_length = 0
        
        # DP to find longest path ending at each vertex with each mask
        dp = [[0] * n for _ in range(1 << n)]
        
        for i in range(n):
            dp[1 << i][i] = 1
        
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)) or dp[mask][last] == 0:
                    continue
                
                max_length = max(max_length, dp[mask][last])
                
                for next_vertex in range(n):
                    if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                        continue
                    
                    new_mask = mask | (1 << next_vertex)
                    dp[new_mask][next_vertex] = max(dp[new_mask][next_vertex], 
                                                   dp[mask][last] + 1)
        
        return max_length
    
    def hamiltonian_path_with_forbidden_vertices(graph, forbidden):
        """Find Hamiltonian path avoiding forbidden vertices"""
        n = len(graph)
        forbidden_set = set(forbidden)
        
        # Create modified graph
        allowed_vertices = [i for i in range(n) if i not in forbidden_set]
        
        if len(allowed_vertices) <= 1:
            return len(allowed_vertices) == 1
        
        # Check if Hamiltonian path exists among allowed vertices
        dp = [[False] * n for _ in range(1 << n)]
        
        for i in allowed_vertices:
            dp[1 << i][i] = True
        
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)) or not dp[mask][last] or last in forbidden_set:
                    continue
                
                for next_vertex in allowed_vertices:
                    if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                        continue
                    
                    new_mask = mask | (1 << next_vertex)
                    dp[new_mask][next_vertex] = True
        
        # Check if all allowed vertices can be visited
        target_mask = 0
        for v in allowed_vertices:
            target_mask |= (1 << v)
        
        return any(dp[target_mask][i] for i in allowed_vertices)
    
    def hamiltonian_paths_starting_from(graph, start_vertex):
        """Count Hamiltonian paths starting from specific vertex"""
        n = len(graph)
        
        dp = [[0] * n for _ in range(1 << n)]
        dp[1 << start_vertex][start_vertex] = 1
        
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)) or dp[mask][last] == 0:
                    continue
                
                for next_vertex in range(n):
                    if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                        continue
                    
                    new_mask = mask | (1 << next_vertex)
                    dp[new_mask][next_vertex] += dp[mask][last]
        
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n))
    
    def minimum_edges_for_hamiltonian_path(n):
        """Minimum number of edges needed for Hamiltonian path to exist"""
        # For a directed graph to guarantee a Hamiltonian path exists,
        # we need at least n-1 edges (like a directed path)
        # But this doesn't guarantee existence - it's a necessary but not sufficient condition
        return n - 1
    
    # Test variants
    test_graphs = [
        # Simple path graph
        [[0, 1, 0],
         [0, 0, 1],
         [0, 0, 0]],
        
        # Cycle
        [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 0, 0, 0]],
        
        # Complete graph (small)
        [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]]
    ]
    
    print("Hamiltonian Path Variants:")
    print("=" * 50)
    
    for i, graph in enumerate(test_graphs):
        n = len(graph)
        print(f"\nGraph {i + 1} (n={n}):")
        
        has_path = has_hamiltonian_path_bitmask_dp(graph)
        path_count = count_hamiltonian_paths(graph)
        longest = longest_simple_path(graph)
        
        print(f"Has Hamiltonian path: {has_path}")
        print(f"Number of Hamiltonian paths: {path_count}")
        print(f"Longest simple path length: {longest}")
        
        # Paths starting from vertex 0
        if n > 0:
            paths_from_0 = hamiltonian_paths_starting_from(graph, 0)
            print(f"Hamiltonian paths from vertex 0: {paths_from_0}")
        
        # Forbidden vertices test
        if n > 2:
            forbidden = [1]
            has_path_forbidden = hamiltonian_path_with_forbidden_vertices(graph, forbidden)
            print(f"Has path avoiding vertex {forbidden}: {has_path_forbidden}")


# Test cases
def test_hamiltonian_path():
    """Test all implementations with various inputs"""
    test_cases = [
        # Simple path
        ([[0, 1, 0], [0, 0, 1], [0, 0, 0]], True),
        
        # Cycle
        ([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], True),
        
        # Complete graph
        ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], True),
        
        # Disconnected
        ([[0, 0, 1], [1, 0, 0], [0, 1, 0]], False),
        
        # Single vertex
        ([[0]], True),
        
        # Two vertices connected
        ([[0, 1], [0, 0]], True),
        
        # Two vertices disconnected
        ([[0, 0], [0, 0]], False)
    ]
    
    print("Testing Hamiltonian Path Solutions:")
    print("=" * 70)
    
    for i, (graph, expected) in enumerate(test_cases):
        n = len(graph)
        print(f"\nTest Case {i + 1} (n={n}):")
        print(f"Expected: {expected}")
        
        # Skip brute force for large graphs
        if n <= 6:
            try:
                brute_force = has_hamiltonian_path_brute_force(graph)
                print(f"Brute Force:      {brute_force} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        bitmask_dp = has_hamiltonian_path_bitmask_dp(graph)
        has_path, path = find_hamiltonian_path_with_reconstruction(graph)
        path_count = count_hamiltonian_paths(graph)
        
        print(f"Bitmask DP:       {bitmask_dp} {'✓' if bitmask_dp == expected else '✗'}")
        print(f"With Path:        {has_path} {'✓' if has_path == expected else '✗'}")
        print(f"Path Count:       {path_count}")
        
        if has_path and path:
            print(f"Example Path:     {path}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    
    example_graph = [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    analysis = hamiltonian_path_analysis(example_graph)
    
    print(f"Graph Analysis:")
    print(f"Vertices: {analysis['num_vertices']}")
    print(f"Edges: {analysis['num_edges']}")
    print(f"Has Hamiltonian path: {analysis['has_hamiltonian_path']}")
    print(f"Number of paths: {analysis['path_count']}")
    
    if analysis['example_path']:
        print(f"Example path: {analysis['example_path']}")
    
    print(f"In-degrees: {analysis['degree_analysis']['in_degrees']}")
    print(f"Out-degrees: {analysis['degree_analysis']['out_degrees']}")
    
    if analysis['impossible_reasons']:
        print(f"Impossible reasons: {analysis['impossible_reasons']}")
    
    if analysis['bottlenecks']:
        print(f"Bottleneck vertices: {analysis['bottlenecks']}")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    hamiltonian_path_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. GRAPH TRAVERSAL: Visit each vertex exactly once without returning")
    print("2. BITMASK STATE: Track visited vertices efficiently")
    print("3. PATH EXISTENCE: NP-complete problem solvable exactly for small graphs")
    print("4. DEGREE ANALYSIS: In/out degree constraints affect path existence")
    print("5. DIRECTED GRAPHS: Direction matters for path construction")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Route Planning: Visit all locations exactly once")
    print("• DNA Sequencing: Reconstruct sequences from fragments")
    print("• Circuit Design: Optimal wire routing in circuits")
    print("• Game Theory: Optimal moves visiting all states")
    print("• Computer Science: Graph traversal and path finding algorithms")


if __name__ == "__main__":
    test_hamiltonian_path()


"""
HAMILTONIAN PATH PROBLEM - COMPLETE GRAPH TRAVERSAL:
====================================================

The Hamiltonian Path Problem represents a fundamental graph traversal challenge:
- Visit each vertex exactly once without returning to start
- Determine existence and count all possible paths
- NP-complete problem with exponential exact solutions
- Foundation for many practical routing and sequencing problems

KEY INSIGHTS:
============
1. **COMPLETE TRAVERSAL**: Visit every vertex exactly once (unlike TSP, no return required)
2. **BITMASK STATE TRACKING**: Efficiently represent visited vertex sets
3. **PATH EXISTENCE**: NP-complete decision problem with exponential solutions
4. **DEGREE CONSTRAINTS**: Graph structure heavily influences path existence
5. **DIRECTED NATURE**: Edge direction critically affects path construction

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n!) time, O(n) space
   - Try all vertex permutations as potential paths
   - Factorial complexity makes it impractical for n > 10

2. **Bitmask DP**: O(n² × 2^n) time, O(n × 2^n) space
   - Dynamic programming with visited vertex bitmasks
   - Optimal exact algorithm for small to medium graphs

3. **Path Reconstruction**: O(n² × 2^n) time, O(n × 2^n) space
   - Include parent tracking for actual path recovery
   - Essential for practical applications

4. **Counting Paths**: O(n² × 2^n) time, O(n × 2^n) space
   - Count all distinct Hamiltonian paths
   - Useful for analyzing graph connectivity

CORE BITMASK DP ALGORITHM:
=========================
```python
def hasHamiltonianPath(graph):
    n = len(graph)
    
    # dp[mask][last] = True if can visit vertices in mask ending at last
    dp = [[False] * n for _ in range(1 << n)]
    
    # Initialize: start from each vertex
    for i in range(n):
        dp[1 << i][i] = True
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or not dp[mask][last]:
                continue
            
            # Try extending to unvisited vertices
            for next_vertex in range(n):
                if (mask & (1 << next_vertex)) or graph[last][next_vertex] == 0:
                    continue
                
                new_mask = mask | (1 << next_vertex)
                dp[new_mask][next_vertex] = True
    
    # Check if any complete path exists
    full_mask = (1 << n) - 1
    return any(dp[full_mask][i] for i in range(n))
```

STATE REPRESENTATION:
====================
**Bitmask Encoding**: Each bit represents whether a vertex has been visited
- `mask = 0001011` → vertices 0, 1, and 3 visited
- `mask = 1111111` → all vertices visited (for n=7)

**DP State**: `dp[mask][last]` = feasibility/count of paths visiting vertices in `mask`, ending at vertex `last`

**State Transitions**: From `(mask, last)` to `(mask | (1 << next), next)` if edge exists

GRAPH STRUCTURE ANALYSIS:
=========================
**Degree Analysis**: Critical for determining path existence
- **Sources**: Vertices with in-degree 0 (potential start points)
- **Sinks**: Vertices with out-degree 0 (potential end points)
- **Isolated**: Vertices with both degrees 0 (impossible to include)

**Necessary Conditions**: For Hamiltonian path existence:
- At most one source vertex
- At most one sink vertex  
- No isolated vertices
- Graph must be "weakly connected"

**Sufficient Conditions**: Much harder to characterize (NP-complete nature)

PATH RECONSTRUCTION:
===================
**Parent Tracking**: Store optimal predecessor for each state
```python
parent[new_mask][next_vertex] = (mask, last_vertex)
```

**Path Recovery**: Backtrack through parent pointers
```python
def reconstruct_path():
    path = []
    mask, vertex = full_mask, end_vertex
    
    while mask != 0:
        path.append(vertex)
        if parent[mask][vertex] == (-1, -1):
            break
        mask, vertex = parent[mask][vertex]
    
    return path[::-1]
```

COUNTING HAMILTONIAN PATHS:
==========================
**Modification**: Change boolean DP to integer counting
```python
dp[mask][last] = number_of_paths_to_reach_state(mask, last)
```

**Applications**: 
- Analyzing graph connectivity richness
- Probabilistic path selection
- Redundancy analysis in networks

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n² × 2^n)
- 2^n possible vertex subsets
- n possible ending vertices per subset
- n transitions to explore per state

**Space Complexity**: O(n × 2^n)
- DP table storage
- Additional space for parent tracking

**Practical Limits**: n ≤ 20-25 depending on available memory

DIRECTED VS UNDIRECTED:
======================
**Directed Graphs**: Edges have direction, more constrained
**Undirected Graphs**: Model as directed with bidirectional edges
**Mixed Graphs**: Some edges directed, others undirected

**Algorithm Adaptation**: Same DP framework works for all variants

APPLICATIONS:
============
- **Route Optimization**: Visit all locations exactly once
- **DNA Sequencing**: Reconstruct genome from overlapping fragments  
- **Circuit Design**: Optimal routing minimizing crossings
- **Compiler Optimization**: Instruction scheduling with dependencies
- **Game AI**: Optimal exploration strategies

RELATED PROBLEMS:
================
- **Traveling Salesman Problem**: Add return-to-start requirement
- **Longest Path Problem**: Find longest simple path in DAG
- **Graph Coloring**: Different type of vertex constraint satisfaction
- **Topological Sorting**: Linear ordering with precedence constraints

VARIANTS:
========
- **Hamiltonian Cycle**: Path that returns to starting vertex
- **Longest Simple Path**: Maximum length path without repeated vertices
- **k-Path Problem**: Path visiting exactly k vertices
- **Forbidden Vertices**: Hamiltonian path avoiding certain vertices

EDGE CASES:
==========
- **Single Vertex**: Trivially has Hamiltonian path
- **No Edges**: Only single vertex case possible
- **Complete Graph**: Always has Hamiltonian path
- **Disconnected Graph**: Generally impossible unless components trivial

OPTIMIZATION TECHNIQUES:
=======================
**Preprocessing**: 
- Remove impossible vertices early
- Identify forced path segments
- Degree-based pruning

**State Pruning**:
- Skip unreachable states
- Use graph structure for early termination

**Memory Optimization**:
- Sparse state representation
- Rolling DP for memory-constrained environments

THEORETICAL SIGNIFICANCE:
========================
**Computational Complexity**: Classic NP-complete problem
**Approximation**: No good polynomial-time approximation exists
**Parameterized Complexity**: Fixed-parameter tractable in graph width
**Structural Analysis**: Reveals fundamental graph connectivity properties

The Hamiltonian Path Problem perfectly demonstrates the power
and limitations of exact exponential algorithms: providing
optimal solutions for moderately-sized instances while
highlighting the need for heuristic approaches in larger
real-world applications.
"""
