"""
LeetCode 847: Shortest Path Visiting All Nodes
Difficulty: Hard
Category: Bitmask DP - Graph Traversal

PROBLEM DESCRIPTION:
===================
You have an undirected, connected graph of n nodes labeled from 0 to n - 1. You are given an array graph where graph[i] is a list of all the nodes connected to node i.

Return the length of the shortest path that visits every node. You may start and end at any node.

Example 1:
Input: graph = [[1,2,3],[0],[0],[0]]
Output: 4
Explanation: One possible path is [1,0,2,0,3]

Example 2:
Input: graph = [[1],[0,2,3],[1],[1]]
Output: 4
Explanation: One possible path is [0,1,3,1,2]

Constraints:
- n == graph.length
- 1 <= n <= 12
- 0 <= graph[i].length < n
- graph[i] does not contain i.
- If graph[a] contains b, then graph[b] contains a.
- The given graph is connected.
"""


def shortest_path_visiting_all_nodes_bfs(graph):
    """
    BFS APPROACH:
    ============
    Use BFS with state (node, visited_mask) to find shortest path.
    
    Time Complexity: O(n * 2^n) - n nodes, 2^n possible visited states
    Space Complexity: O(n * 2^n) - queue and visited set
    """
    n = len(graph)
    if n == 1:
        return 0
    
    from collections import deque
    
    # State: (current_node, visited_mask, path_length)
    queue = deque()
    visited = set()
    
    # Start from each node
    for start in range(n):
        initial_mask = 1 << start
        queue.append((start, initial_mask, 0))
        visited.add((start, initial_mask))
    
    target_mask = (1 << n) - 1  # All nodes visited
    
    while queue:
        node, mask, length = queue.popleft()
        
        # Try visiting each neighbor
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            new_state = (neighbor, new_mask)
            
            if new_mask == target_mask:
                return length + 1
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((neighbor, new_mask, length + 1))
    
    return -1  # Should never reach here for connected graph


def shortest_path_visiting_all_nodes_dp(graph):
    """
    DYNAMIC PROGRAMMING APPROACH:
    ============================
    Use DP with bitmask to track visited nodes.
    
    Time Complexity: O(n^2 * 2^n) - n^2 transitions, 2^n states
    Space Complexity: O(n * 2^n) - DP table
    """
    n = len(graph)
    if n == 1:
        return 0
    
    # dp[mask][node] = minimum steps to reach node with visited set = mask
    dp = [[float('inf')] * n for _ in range(1 << n)]
    
    # Initialize: start from each node
    for i in range(n):
        dp[1 << i][i] = 0
    
    for mask in range(1 << n):
        for node in range(n):
            if dp[mask][node] == float('inf'):
                continue
            
            # Try visiting each neighbor
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                dp[new_mask][neighbor] = min(dp[new_mask][neighbor], 
                                           dp[mask][node] + 1)
    
    # Find minimum among all nodes with all visited
    target_mask = (1 << n) - 1
    return min(dp[target_mask])


def shortest_path_visiting_all_nodes_optimized_dp(graph):
    """
    OPTIMIZED DP APPROACH:
    =====================
    Use space-optimized DP with better state transitions.
    
    Time Complexity: O(n^2 * 2^n) - optimal for this problem
    Space Complexity: O(n * 2^n) - DP table
    """
    n = len(graph)
    if n == 1:
        return 0
    
    # dp[mask][node] = minimum cost to visit all nodes in mask ending at node
    dp = [[float('inf')] * n for _ in range(1 << n)]
    
    # Base case: start from any single node
    for i in range(n):
        dp[1 << i][i] = 0
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for node in range(n):
            if not (mask & (1 << node)):
                continue
            
            # Try coming from each neighbor
            for neighbor in graph[node]:
                prev_mask = mask ^ (1 << node)  # Remove current node
                if prev_mask & (1 << neighbor):  # Neighbor was visited
                    dp[mask][node] = min(dp[mask][node], 
                                        dp[prev_mask][neighbor] + 1)
    
    # Answer: minimum cost with all nodes visited
    full_mask = (1 << n) - 1
    return min(dp[full_mask])


def shortest_path_visiting_all_nodes_bitmask_bfs(graph):
    """
    BITMASK BFS APPROACH:
    ====================
    Combine bitmask state tracking with BFS for optimal solution.
    
    Time Complexity: O(n * 2^n) - each state visited once
    Space Complexity: O(n * 2^n) - queue and visited tracking
    """
    from collections import deque
    
    n = len(graph)
    if n == 1:
        return 0
    
    # State: (node, visited_mask)
    queue = deque()
    visited = set()
    
    # Initialize with each node as starting point
    for start in range(n):
        state = (start, 1 << start)
        queue.append((state, 0))  # (state, distance)
        visited.add(state)
    
    target = (1 << n) - 1  # All nodes visited
    
    while queue:
        (node, mask), dist = queue.popleft()
        
        if mask == target:
            return dist
        
        # Explore neighbors
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            new_state = (neighbor, new_mask)
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, dist + 1))
    
    return -1


def shortest_path_visiting_all_nodes_with_path(graph):
    """
    BITMASK DP WITH PATH RECONSTRUCTION:
    ===================================
    Find shortest path and reconstruct the actual path.
    
    Time Complexity: O(n^2 * 2^n) - DP computation + path reconstruction
    Space Complexity: O(n * 2^n) - DP table + parent tracking
    """
    n = len(graph)
    if n == 1:
        return 0, [0]
    
    # dp[mask][node] = (min_cost, parent_state)
    dp = [[(float('inf'), None)] * n for _ in range(1 << n)]
    
    # Initialize starting states
    for i in range(n):
        dp[1 << i][i] = (0, None)
    
    # Fill DP table with parent tracking
    for mask in range(1, 1 << n):
        for node in range(n):
            if not (mask & (1 << node)):
                continue
            
            for neighbor in graph[node]:
                prev_mask = mask ^ (1 << node)
                if prev_mask & (1 << neighbor):
                    new_cost = dp[prev_mask][neighbor][0] + 1
                    if new_cost < dp[mask][node][0]:
                        dp[mask][node] = (new_cost, (prev_mask, neighbor))
    
    # Find optimal ending state
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    best_end = -1
    
    for node in range(n):
        if dp[full_mask][node][0] < min_cost:
            min_cost = dp[full_mask][node][0]
            best_end = node
    
    # Reconstruct path
    path = []
    current_mask, current_node = full_mask, best_end
    
    while current_mask != 0:
        path.append(current_node)
        parent_info = dp[current_mask][current_node][1]
        
        if parent_info is None:
            break
        
        current_mask, current_node = parent_info
    
    path.reverse()
    return min_cost, path


def shortest_path_visiting_all_nodes_analysis(graph):
    """
    COMPREHENSIVE ANALYSIS:
    ======================
    Analyze the shortest path problem with detailed insights.
    """
    n = len(graph)
    print(f"Shortest Path Visiting All Nodes Analysis:")
    print(f"Number of nodes: {n}")
    print(f"Graph adjacency list: {graph}")
    
    # Graph properties
    total_edges = sum(len(neighbors) for neighbors in graph) // 2
    print(f"Number of edges: {total_edges}")
    print(f"Average degree: {2 * total_edges / n:.2f}")
    
    # Connectivity analysis
    degrees = [len(neighbors) for neighbors in graph]
    print(f"Node degrees: {degrees}")
    print(f"Min degree: {min(degrees)}, Max degree: {max(degrees)}")
    
    # Different approaches
    bfs_result = shortest_path_visiting_all_nodes_bfs(graph)
    dp_result = shortest_path_visiting_all_nodes_dp(graph)
    optimized_result = shortest_path_visiting_all_nodes_optimized_dp(graph)
    bitmask_bfs_result = shortest_path_visiting_all_nodes_bitmask_bfs(graph)
    
    print(f"\nResults:")
    print(f"BFS approach: {bfs_result}")
    print(f"DP approach: {dp_result}")
    print(f"Optimized DP: {optimized_result}")
    print(f"Bitmask BFS: {bitmask_bfs_result}")
    
    # Path reconstruction
    if n <= 8:
        min_cost, path = shortest_path_visiting_all_nodes_with_path(graph)
        print(f"\nOptimal path length: {min_cost}")
        print(f"One optimal path: {path}")
        
        # Verify path
        if len(path) > 1:
            visited_nodes = set()
            valid_path = True
            for i in range(len(path)):
                visited_nodes.add(path[i])
                if i > 0 and path[i] not in graph[path[i-1]]:
                    valid_path = False
                    break
            
            print(f"Path validity: {'Valid' if valid_path else 'Invalid'}")
            print(f"Nodes visited: {len(visited_nodes)}/{n}")
    
    # State space analysis
    total_states = n * (2 ** n)
    print(f"\nState space analysis:")
    print(f"Total possible states: {total_states}")
    print(f"States per node: {2 ** n}")
    print(f"Memory requirement: ~{total_states * 4 / 1024:.1f} KB")
    
    return bitmask_bfs_result


def shortest_path_visiting_all_nodes_variants():
    """
    SHORTEST PATH VARIANTS:
    ======================
    Different scenarios and modifications.
    """
    
    def shortest_path_with_start_end(graph, start, end):
        """Find shortest path from start to end visiting all nodes"""
        n = len(graph)
        from collections import deque
        
        queue = deque([(start, 1 << start, 0)])
        visited = set([(start, 1 << start)])
        target = (1 << n) - 1
        
        while queue:
            node, mask, dist = queue.popleft()
            
            if mask == target and node == end:
                return dist
            
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                state = (neighbor, new_mask)
                
                if state not in visited:
                    visited.add(state)
                    queue.append((neighbor, new_mask, dist + 1))
        
        return -1
    
    def shortest_path_avoiding_node(graph, avoid_node):
        """Find shortest path visiting all nodes except avoid_node"""
        n = len(graph)
        if avoid_node >= n:
            return shortest_path_visiting_all_nodes_bitmask_bfs(graph)
        
        # Create modified graph without avoid_node
        modified_graph = []
        node_mapping = []
        
        for i in range(n):
            if i != avoid_node:
                node_mapping.append(i)
                new_neighbors = []
                for neighbor in graph[i]:
                    if neighbor != avoid_node:
                        # Map to new index
                        new_idx = node_mapping.index(neighbor) if neighbor in node_mapping else len(node_mapping)
                        if neighbor in node_mapping:
                            new_neighbors.append(node_mapping.index(neighbor))
                
                modified_graph.append(new_neighbors)
        
        return shortest_path_visiting_all_nodes_bitmask_bfs(modified_graph)
    
    def count_shortest_paths(graph):
        """Count number of shortest paths visiting all nodes"""
        n = len(graph)
        from collections import deque
        
        # Find shortest path length first
        min_length = shortest_path_visiting_all_nodes_bitmask_bfs(graph)
        
        # Count paths of this length
        queue = deque()
        visited = {}
        
        for start in range(n):
            state = (start, 1 << start)
            queue.append((state, 0))
            visited[state] = 1
        
        target = (1 << n) - 1
        count = 0
        
        while queue:
            (node, mask), dist = queue.popleft()
            
            if dist > min_length:
                break
            
            if mask == target and dist == min_length:
                count += visited[(node, mask)]
                continue
            
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                new_state = (neighbor, new_mask)
                
                if new_state not in visited:
                    visited[new_state] = 0
                
                if dist + 1 <= min_length:
                    visited[new_state] += visited[(node, mask)]
                    queue.append((new_state, dist + 1))
        
        return count
    
    def longest_path_visiting_all_nodes(graph):
        """Find longest simple path visiting all nodes"""
        # This is much more complex - simplified approximation
        n = len(graph)
        if n <= 4:
            # For small graphs, try all permutations
            from itertools import permutations
            max_length = 0
            
            for perm in permutations(range(n)):
                length = 0
                valid = True
                for i in range(len(perm) - 1):
                    if perm[i+1] not in graph[perm[i]]:
                        valid = False
                        break
                    length += 1
                
                if valid:
                    max_length = max(max_length, length)
            
            return max_length
        
        return -1  # Too complex for larger graphs
    
    # Test variants
    test_graphs = [
        [[1, 2, 3], [0], [0], [0]],
        [[1], [0, 2, 3], [1], [1]],
        [[1, 2], [0, 3], [0, 3], [1, 2]],
        [[1, 3], [0, 2], [1, 3], [0, 2]]
    ]
    
    print("Shortest Path Variants:")
    print("=" * 50)
    
    for i, graph in enumerate(test_graphs):
        n = len(graph)
        print(f"\nGraph {i+1}: {graph}")
        
        basic_result = shortest_path_visiting_all_nodes_bitmask_bfs(graph)
        print(f"Basic shortest path: {basic_result}")
        
        # Start-end variants
        if n >= 2:
            start_end_result = shortest_path_with_start_end(graph, 0, n-1)
            print(f"Path from 0 to {n-1}: {start_end_result}")
        
        # Avoiding node
        if n > 2:
            avoid_result = shortest_path_avoiding_node(graph, 0)
            print(f"Avoiding node 0: {avoid_result}")
        
        # Count shortest paths
        if n <= 6:
            path_count = count_shortest_paths(graph)
            print(f"Number of shortest paths: {path_count}")
        
        # Longest path (for small graphs)
        if n <= 4:
            longest = longest_path_visiting_all_nodes(graph)
            print(f"Longest simple path: {longest}")


# Test cases
def test_shortest_path_visiting_all_nodes():
    """Test all implementations with various inputs"""
    test_cases = [
        ([[1,2,3],[0],[0],[0]], 4),
        ([[1],[0,2,3],[1],[1]], 4),
        ([[1,2],[0,3],[0,3],[1,2]], 4),
        ([[1,3],[0,2],[1,3],[0,2]], 4),
        ([[]], 0),
        ([[1],[0]], 1),
        ([[1,2],[0,2],[0,1]], 2),
        ([[2,3],[2,3],[0,1],[0,1]], 3)
    ]
    
    print("Testing Shortest Path Visiting All Nodes Solutions:")
    print("=" * 70)
    
    for i, (graph, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"graph = {graph}")
        print(f"Expected: {expected}")
        
        bfs = shortest_path_visiting_all_nodes_bfs(graph)
        dp = shortest_path_visiting_all_nodes_dp(graph)
        optimized = shortest_path_visiting_all_nodes_optimized_dp(graph)
        bitmask_bfs = shortest_path_visiting_all_nodes_bitmask_bfs(graph)
        
        print(f"BFS:              {bfs:>4} {'✓' if bfs == expected else '✗'}")
        print(f"DP:               {dp:>4} {'✓' if dp == expected else '✗'}")
        print(f"Optimized DP:     {optimized:>4} {'✓' if optimized == expected else '✗'}")
        print(f"Bitmask BFS:      {bitmask_bfs:>4} {'✓' if bitmask_bfs == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    shortest_path_visiting_all_nodes_analysis([[1,2,3],[0],[0],[0]])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    shortest_path_visiting_all_nodes_variants()
    
    # Performance comparison
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON:")
    performance_graphs = [
        [[i for i in range(6) if i != j] for j in range(6)],  # Complete graph
        [[1, 2], [0, 3], [0, 4], [1, 5], [2, 5], [3, 4]],    # Path-like
        [[1, 3], [0, 2, 4], [1, 5], [0, 4], [1, 3, 5], [2, 4]]  # Complex
    ]
    
    for i, graph in enumerate(performance_graphs):
        print(f"\nGraph {i+1} (n={len(graph)}):")
        
        import time
        
        start = time.time()
        bfs_result = shortest_path_visiting_all_nodes_bfs(graph)
        bfs_time = time.time() - start
        
        start = time.time()
        dp_result = shortest_path_visiting_all_nodes_optimized_dp(graph)
        dp_time = time.time() - start
        
        print(f"BFS: {bfs_result} (Time: {bfs_time:.6f}s)")
        print(f"DP:  {dp_result} (Time: {dp_time:.6f}s)")
        print(f"Match: {'✓' if bfs_result == dp_result else '✗'}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. BITMASK STATE: Represent visited nodes with bit patterns")
    print("2. MULTI-SOURCE BFS: Start from all nodes simultaneously")
    print("3. STATE SPACE: O(n * 2^n) possible (node, visited_set) states")
    print("4. OPTIMAL SUBSTRUCTURE: Shortest path to state is optimal")
    print("5. GRAPH TRAVERSAL: Combine path finding with complete coverage")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Network Coverage: Visit all network nodes efficiently")
    print("• Delivery Routes: Optimal routes covering all destinations")
    print("• Graph Algorithms: Hamiltonian path variations")
    print("• Traveling Salesman: Foundation for TSP-like problems")
    print("• Computer Science: State space search and graph traversal")


if __name__ == "__main__":
    test_shortest_path_visiting_all_nodes()


"""
SHORTEST PATH VISITING ALL NODES - GRAPH TRAVERSAL WITH BITMASKS:
=================================================================

This problem demonstrates advanced Bitmask DP for graph traversal:
- State compression for visited node tracking
- Multi-source shortest path algorithms
- Optimal substructure in graph problems
- Integration of BFS/DP with bitmask state management

KEY INSIGHTS:
============
1. **BITMASK STATE REPRESENTATION**: Use bits to track which nodes have been visited
2. **MULTI-SOURCE INITIALIZATION**: Start BFS/DP from all possible starting nodes
3. **STATE SPACE OPTIMIZATION**: Combine current node with visited set for efficient exploration
4. **SHORTEST PATH GUARANTEE**: BFS ensures optimal path length discovery
5. **COMPLETE COVERAGE**: Ensure all nodes are visited exactly once

ALGORITHM APPROACHES:
====================

1. **BFS with Bitmask**: O(n × 2^n) time, O(n × 2^n) space
   - Multi-source BFS with state (node, visited_mask)
   - Guarantees shortest path through level-order exploration

2. **Dynamic Programming**: O(n² × 2^n) time, O(n × 2^n) space
   - Bottom-up DP building optimal paths incrementally
   - Better for understanding state transitions

3. **Optimized DP**: O(n² × 2^n) time, O(n × 2^n) space
   - Streamlined state transitions with cleaner implementation
   - Space-optimized version possible

4. **Bitmask BFS**: O(n × 2^n) time, O(n × 2^n) space
   - Clean BFS implementation with bitmask state tracking
   - Most practical approach for this problem

CORE BITMASK BFS ALGORITHM:
==========================
```python
def shortestPathLength(graph):
    from collections import deque
    
    n = len(graph)
    if n == 1:
        return 0
    
    # State: (node, visited_mask)
    queue = deque()
    visited = set()
    
    # Multi-source initialization: start from each node
    for start in range(n):
        state = (start, 1 << start)
        queue.append((state, 0))
        visited.add(state)
    
    target = (1 << n) - 1  # All nodes visited
    
    while queue:
        (node, mask), dist = queue.popleft()
        
        if mask == target:
            return dist
        
        # Explore all neighbors
        for neighbor in graph[node]:
            new_mask = mask | (1 << neighbor)
            new_state = (neighbor, new_mask)
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, dist + 1))
    
    return -1
```

MULTI-SOURCE BFS STRATEGY:
=========================
**Initialization**: Start from every node simultaneously
- Each starting node represents a different potential beginning
- All have equal opportunity to find the optimal solution
- Eliminates need to try each start node separately

**State Management**: `(current_node, visited_bitmask)`
- Current node: where we are now
- Visited bitmask: which nodes we've seen so far
- Distance: implicit through BFS level

**Optimality**: BFS guarantees shortest path
- Level-order exploration ensures minimum distance
- First time we visit all nodes gives optimal result

BITMASK OPERATIONS FOR GRAPH TRAVERSAL:
======================================
**Initialize Single Node**: `mask = 1 << start_node`
**Add Node to Visited**: `new_mask = mask | (1 << neighbor)`
**Check All Visited**: `mask == (1 << n) - 1`
**Count Visited Nodes**: `bin(mask).count('1')`

**State Transitions**:
```python
for neighbor in graph[node]:
    new_mask = mask | (1 << neighbor)  # Visit neighbor
    new_state = (neighbor, new_mask)
```

DYNAMIC PROGRAMMING FORMULATION:
===============================
**State Definition**: `dp[mask][node]` = minimum steps to reach `node` with visited set `mask`

**Base Cases**: `dp[1 << i][i] = 0` for all starting nodes

**Transitions**:
```python
for mask in range(1 << n):
    for node in range(n):
        if dp[mask][node] < infinity:
            for neighbor in graph[node]:
                new_mask = mask | (1 << neighbor)
                dp[new_mask][neighbor] = min(dp[new_mask][neighbor], 
                                           dp[mask][node] + 1)
```

**Answer**: `min(dp[(1 << n) - 1][i] for i in range(n))`

STATE SPACE ANALYSIS:
====================
**Total States**: n × 2^n
- n possible current nodes
- 2^n possible visited sets
- Each state represents unique configuration

**Memory Requirements**: O(n × 2^n)
- Practical limit: n ≤ 15-20 due to exponential growth
- For n=12: ~50K states, manageable
- For n=20: ~20M states, challenging

**State Pruning**: Many states unreachable
- Connected graph constraints reduce actual state space
- Graph structure affects exploration efficiency

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n × 2^n)
- Each state visited at most once
- Each state explores up to n neighbors
- BFS guarantees optimal visiting order

**Space Complexity**: O(n × 2^n)
- Queue storage: O(n × 2^n) in worst case
- Visited set: O(n × 2^n) states
- Graph storage: O(n²) negligible by comparison

OPTIMIZATION TECHNIQUES:
=======================
**Early Termination**: Stop when all nodes visited
**State Deduplication**: Use visited set to avoid cycles
**Multi-Source Start**: Parallel exploration from all nodes
**Memory Management**: Use efficient data structures

**Graph-Specific Optimizations**:
- Sparse graphs: adjacency list representation
- Dense graphs: might benefit from different approaches
- Special structures: trees, cycles have specific optimizations

APPLICATIONS:
============
- **Network Traversal**: Visit all network nodes with minimum hops
- **Delivery Optimization**: Shortest route covering all destinations
- **Graph Algorithms**: Foundation for Hamiltonian path problems
- **Traveling Salesman**: Simplified TSP without return requirement
- **Coverage Problems**: Minimum time to cover all locations

RELATED PROBLEMS:
================
- **Traveling Salesman Problem**: Return to start node required
- **Hamiltonian Path**: Visit each node exactly once
- **Minimum Spanning Tree**: Connect all nodes with minimum cost
- **Graph Coloring**: Different type of graph coverage problem

VARIANTS:
========
- **Fixed Start/End**: Constrain starting and ending nodes
- **Avoid Nodes**: Skip certain nodes during traversal
- **Weighted Graphs**: Consider edge weights in path calculation
- **Multiple Visits**: Allow revisiting nodes with different constraints

EDGE CASES:
==========
- **Single Node**: Path length 0
- **Two Nodes**: Path length 1 if connected
- **Disconnected Graph**: No solution exists
- **Complete Graph**: Many optimal paths exist

PRACTICAL CONSIDERATIONS:
========================
**Scalability**: Limited to small graphs (n ≤ 15-20)
**Memory Usage**: Exponential space requirements
**Implementation**: BFS generally simpler than DP
**Performance**: Graph structure significantly affects runtime

This problem beautifully demonstrates how Bitmask DP can solve
complex graph traversal problems by efficiently encoding and
exploring the exponential state space of visited node combinations,
while BFS ensures optimal path discovery.
"""
