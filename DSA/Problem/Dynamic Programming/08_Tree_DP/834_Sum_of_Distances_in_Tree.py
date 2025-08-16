"""
LeetCode 834: Sum of Distances in Tree
Difficulty: Hard
Category: Tree DP - Re-rooting Technique

PROBLEM DESCRIPTION:
===================
There is an undirected connected tree with n nodes labeled from 0 to n - 1 and n - 1 edges.

You are given the integer n and the array edges where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.

Return an array answer of length n where answer[i] is the sum of the distances between node i and all other nodes in the tree.

Example 1:
Input: n = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
Output: [8,12,6,10,10,10]
Explanation: The tree is shown above.
We can see that dist(0,1) + dist(0,2) + dist(0,3) + dist(0,4) + dist(0,5)
= 1 + 1 + 2 + 2 + 2 = 8.
Hence, answer[0] = 8, and so on.

Example 2:
Input: n = 1, edges = []
Output: [0]

Example 3:
Input: n = 2, edges = [[1,0]]
Output: [1,1]

Constraints:
- 1 <= n <= 3 * 10^4
- edges.length == n - 1
- edges[i].length == 2
- 0 <= ai, bi < n
- ai != bi
- The given input represents a valid tree.
"""

from collections import defaultdict, deque


def sum_of_distances_brute_force(n, edges):
    """
    BRUTE FORCE APPROACH:
    ====================
    For each node, run BFS to calculate distances to all other nodes.
    
    Time Complexity: O(n^2) - BFS from each node
    Space Complexity: O(n) - graph storage + BFS queue
    """
    if n == 1:
        return [0]
    
    # Build adjacency list
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    result = []
    
    for start in range(n):
        # BFS from start node
        queue = deque([start])
        visited = {start}
        distance_sum = 0
        distance = 0
        
        while queue:
            level_size = len(queue)
            distance_sum += distance * level_size
            distance += 1
            
            for _ in range(level_size):
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        result.append(distance_sum)
    
    return result


def sum_of_distances_tree_dp(n, edges):
    """
    TREE DP WITH RE-ROOTING:
    =======================
    Two-pass algorithm: first calculate for root, then re-root for all nodes.
    
    Time Complexity: O(n) - two DFS passes
    Space Complexity: O(n) - graph storage + recursion
    """
    if n == 1:
        return [0]
    
    # Build adjacency list
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    # First pass: calculate subtree sizes and distances for root 0
    subtree_size = [0] * n
    dist_sum = [0] * n
    
    def dfs1(node, parent):
        subtree_size[node] = 1
        for child in graph[node]:
            if child != parent:
                dfs1(child, node)
                subtree_size[node] += subtree_size[child]
                dist_sum[node] += dist_sum[child] + subtree_size[child]
    
    # Second pass: re-root to calculate answer for all nodes
    def dfs2(node, parent):
        for child in graph[node]:
            if child != parent:
                # Re-root from node to child
                # Remove child subtree contribution
                dist_sum[child] = dist_sum[node] - subtree_size[child]
                # Add contribution from nodes not in child subtree
                dist_sum[child] += (n - subtree_size[child])
                dfs2(child, node)
    
    dfs1(0, -1)
    dfs2(0, -1)
    
    return dist_sum


def sum_of_distances_detailed(n, edges):
    """
    DETAILED RE-ROOTING WITH STEP TRACKING:
    ======================================
    Track the re-rooting process with detailed calculations.
    
    Time Complexity: O(n) - two DFS passes
    Space Complexity: O(n) - additional tracking arrays
    """
    if n == 1:
        return [0], {"process": "Single node, no distances"}
    
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    subtree_size = [0] * n
    dist_sum = [0] * n
    process_log = []
    
    def dfs1(node, parent, depth=0):
        process_log.append(f"DFS1: Visiting node {node} at depth {depth}")
        subtree_size[node] = 1
        
        for child in graph[node]:
            if child != parent:
                dfs1(child, node, depth + 1)
                subtree_size[node] += subtree_size[child]
                dist_sum[node] += dist_sum[child] + subtree_size[child]
                process_log.append(f"DFS1: Node {node} updated - subtree_size: {subtree_size[node]}, dist_sum: {dist_sum[node]}")
    
    def dfs2(node, parent):
        process_log.append(f"DFS2: Re-rooting at node {node}")
        
        for child in graph[node]:
            if child != parent:
                old_child_dist = dist_sum[child]
                
                # Re-root calculation
                dist_sum[child] = dist_sum[node] - subtree_size[child] + (n - subtree_size[child])
                
                process_log.append(f"DFS2: Child {child} - old_dist: {old_child_dist}, new_dist: {dist_sum[child]}")
                process_log.append(f"      Formula: {dist_sum[node]} - {subtree_size[child]} + {n - subtree_size[child]} = {dist_sum[child]}")
                
                dfs2(child, node)
    
    process_log.append("Starting DFS1 from root 0")
    dfs1(0, -1)
    
    process_log.append(f"After DFS1 - Root 0 distance sum: {dist_sum[0]}")
    process_log.append("Starting DFS2 for re-rooting")
    dfs2(0, -1)
    
    return dist_sum, {"process_log": process_log, "subtree_sizes": subtree_size}


def sum_of_distances_with_path_analysis(n, edges):
    """
    DISTANCE ANALYSIS WITH PATH DETAILS:
    ===================================
    Analyze distance patterns and provide detailed insights.
    
    Time Complexity: O(n) - optimized tree DP
    Space Complexity: O(n) - additional analysis storage
    """
    if n == 1:
        return [0]
    
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    subtree_size = [0] * n
    dist_sum = [0] * n
    depth = [0] * n
    parent = [-1] * n
    
    # Enhanced DFS1 with additional information
    def dfs1(node, par, d):
        parent[node] = par
        depth[node] = d
        subtree_size[node] = 1
        
        for child in graph[node]:
            if child != par:
                dfs1(child, node, d + 1)
                subtree_size[node] += subtree_size[child]
                dist_sum[node] += dist_sum[child] + subtree_size[child]
    
    def dfs2(node, par):
        for child in graph[node]:
            if child != par:
                dist_sum[child] = dist_sum[node] - subtree_size[child] + (n - subtree_size[child])
                dfs2(child, node)
    
    dfs1(0, -1, 0)
    dfs2(0, -1)
    
    # Analysis
    analysis = {
        'min_distance_sum': min(dist_sum),
        'max_distance_sum': max(dist_sum),
        'optimal_root': dist_sum.index(min(dist_sum)),
        'tree_diameter': max(depth) * 2,  # Approximation
        'average_distance_sum': sum(dist_sum) / n,
        'subtree_sizes': subtree_size,
        'depths_from_root0': depth
    }
    
    return dist_sum, analysis


def sum_of_distances_iterative(n, edges):
    """
    ITERATIVE IMPLEMENTATION:
    ========================
    Stack-based implementation of the re-rooting technique.
    
    Time Complexity: O(n) - two iterative passes
    Space Complexity: O(n) - stack and arrays
    """
    if n == 1:
        return [0]
    
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    subtree_size = [0] * n
    dist_sum = [0] * n
    
    # First pass: calculate subtree sizes and distances (post-order)
    stack = [(0, -1, False)]  # (node, parent, processed)
    
    while stack:
        node, parent, processed = stack.pop()
        
        if processed:
            subtree_size[node] = 1
            for child in graph[node]:
                if child != parent:
                    subtree_size[node] += subtree_size[child]
                    dist_sum[node] += dist_sum[child] + subtree_size[child]
        else:
            stack.append((node, parent, True))
            for child in graph[node]:
                if child != parent:
                    stack.append((child, node, False))
    
    # Second pass: re-root (pre-order)
    stack = [(0, -1)]
    
    while stack:
        node, parent = stack.pop()
        
        for child in graph[node]:
            if child != parent:
                dist_sum[child] = dist_sum[node] - subtree_size[child] + (n - subtree_size[child])
                stack.append((child, node))
    
    return dist_sum


def sum_of_distances_analysis(n, edges):
    """
    COMPREHENSIVE TREE DISTANCE ANALYSIS:
    ====================================
    Analyze tree structure and distance patterns.
    """
    if n == 1:
        print("Single node tree - all distances are 0")
        return [0]
    
    # Build graph and analyze structure
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    print(f"Tree Analysis for {n} nodes:")
    print(f"Number of edges: {len(edges)}")
    
    # Degree analysis
    degrees = [len(graph[i]) for i in range(n)]
    print(f"Degree distribution: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/n:.2f}")
    print(f"Leaves (degree 1): {degrees.count(1)}")
    print(f"Internal nodes: {n - degrees.count(1)}")
    
    # Calculate distances with analysis
    dist_sums, analysis = sum_of_distances_with_path_analysis(n, edges)
    
    print(f"\nDistance Analysis:")
    print(f"Minimum distance sum: {analysis['min_distance_sum']} (node {analysis['optimal_root']})")
    print(f"Maximum distance sum: {analysis['max_distance_sum']}")
    print(f"Average distance sum: {analysis['average_distance_sum']:.2f}")
    print(f"Range: {analysis['max_distance_sum'] - analysis['min_distance_sum']}")
    
    # Tree structure insights
    print(f"\nTree Structure:")
    print(f"Tree diameter (approx): {analysis['tree_diameter']}")
    print(f"Root 0 depth distribution: {dict([(d, analysis['depths_from_root0'].count(d)) for d in range(max(analysis['depths_from_root0']) + 1)])}")
    
    # Optimal root analysis
    optimal_root = analysis['optimal_root']
    print(f"\nOptimal Root Analysis:")
    print(f"Best root for minimum total distance: node {optimal_root}")
    print(f"Improvement over root 0: {dist_sums[0] - dist_sums[optimal_root]}")
    
    return dist_sums


def sum_of_distances_variants():
    """
    TREE DISTANCE VARIANTS:
    ======================
    Different scenarios and modifications.
    """
    
    def sum_of_weighted_distances(n, edges, weights):
        """Sum of distances with weighted nodes"""
        if n == 1:
            return [0]
        
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        subtree_weight = [0] * n
        weighted_dist_sum = [0] * n
        
        def dfs1(node, parent):
            subtree_weight[node] = weights[node]
            for child in graph[node]:
                if child != parent:
                    dfs1(child, node)
                    subtree_weight[node] += subtree_weight[child]
                    weighted_dist_sum[node] += weighted_dist_sum[child] + subtree_weight[child]
        
        def dfs2(node, parent):
            for child in graph[node]:
                if child != parent:
                    total_other_weight = sum(weights) - subtree_weight[child]
                    weighted_dist_sum[child] = (weighted_dist_sum[node] - 
                                               weighted_dist_sum[child] - 
                                               subtree_weight[child] + 
                                               total_other_weight)
                    dfs2(child, node)
        
        dfs1(0, -1)
        dfs2(0, -1)
        return weighted_dist_sum
    
    def sum_of_distances_max_depth(n, edges, max_depth):
        """Sum of distances up to maximum depth"""
        if n == 1:
            return [0]
        
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        def calculate_limited_distances(root):
            total_dist = 0
            queue = deque([(root, 0)])
            visited = {root}
            
            while queue:
                node, dist = queue.popleft()
                
                if dist <= max_depth:
                    total_dist += dist
                    
                    if dist < max_depth:
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append((neighbor, dist + 1))
            
            return total_dist
        
        return [calculate_limited_distances(i) for i in range(n)]
    
    def find_tree_center(n, edges):
        """Find the center(s) of the tree"""
        if n == 1:
            return [0]
        
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        # Calculate distance sums
        dist_sums = sum_of_distances_tree_dp(n, edges)
        min_dist_sum = min(dist_sums)
        
        # Return all nodes with minimum distance sum
        centers = [i for i in range(n) if dist_sums[i] == min_dist_sum]
        return centers
    
    # Test with sample tree
    sample_n = 6
    sample_edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
    
    print("Tree Distance Variants:")
    print("=" * 50)
    
    basic_distances = sum_of_distances_tree_dp(sample_n, sample_edges)
    print(f"Basic distance sums: {basic_distances}")
    
    weights = [1, 2, 1, 1, 1, 1]
    weighted_distances = sum_of_weighted_distances(sample_n, sample_edges, weights)
    print(f"Weighted distance sums: {weighted_distances}")
    
    limited_distances = sum_of_distances_max_depth(sample_n, sample_edges, 2)
    print(f"Distance sums (max depth 2): {limited_distances}")
    
    centers = find_tree_center(sample_n, sample_edges)
    print(f"Tree center(s): {centers}")


# Test cases
def test_sum_of_distances():
    """Test all implementations with various tree configurations"""
    
    test_cases = [
        (6, [[0,1],[0,2],[2,3],[2,4],[2,5]], [8,12,6,10,10,10]),
        (1, [], [0]),
        (2, [[1,0]], [1,1]),
        (3, [[0,1],[1,2]], [3,2,3]),
        (4, [[0,1],[1,2],[1,3]], [6,3,5,5]),
        (5, [[0,1],[0,2],[0,3],[0,4]], [4,7,7,7,7])
    ]
    
    print("Testing Sum of Distances in Tree Solutions:")
    print("=" * 70)
    
    for i, (n, edges, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}, edges = {edges}")
        print(f"Expected: {expected}")
        
        # Skip brute force for larger inputs
        if n <= 10:
            brute_force = sum_of_distances_brute_force(n, edges)
            print(f"Brute Force:      {brute_force} {'✓' if brute_force == expected else '✗'}")
        
        tree_dp = sum_of_distances_tree_dp(n, edges)
        iterative = sum_of_distances_iterative(n, edges)
        
        print(f"Tree DP:          {tree_dp} {'✓' if tree_dp == expected else '✗'}")
        print(f"Iterative:        {iterative} {'✓' if iterative == expected else '✗'}")
        
        # Show detailed analysis for interesting cases
        if n <= 8:
            detailed_result, details = sum_of_distances_detailed(n, edges)
            print(f"Subtree sizes: {details['subtree_sizes']}")
    
    # Comprehensive analysis example
    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS EXAMPLE:")
    print("-" * 40)
    sum_of_distances_analysis(6, [[0,1],[0,2],[2,3],[2,4],[2,5]])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    sum_of_distances_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. RE-ROOTING TECHNIQUE: Calculate for one root, then adjust for all others")
    print("2. TWO-PASS ALGORITHM: First pass for subtree info, second for re-rooting")
    print("3. SUBTREE SIZE TRACKING: Essential for efficient distance calculations")
    print("4. LINEAR TIME: O(n) solution vs O(n^2) brute force")
    print("5. TREE CENTER: Node(s) with minimum sum of distances")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Facility Location: Optimal placement to minimize total distances")
    print("• Network Design: Central hub placement in tree networks")
    print("• Logistics: Warehouse location optimization")
    print("• Graph Theory: Tree center and diameter calculations")
    print("• Distributed Systems: Optimal coordinator placement")


if __name__ == "__main__":
    test_sum_of_distances()


"""
SUM OF DISTANCES IN TREE - ADVANCED RE-ROOTING TECHNIQUE:
=========================================================

This problem showcases the powerful re-rooting technique in Tree DP:
- Efficient calculation of distance sums from all possible roots
- Two-pass algorithm reducing O(n²) to O(n) complexity
- Subtree information propagation and re-calculation
- Advanced tree analysis and optimization applications

KEY INSIGHTS:
============
1. **RE-ROOTING TECHNIQUE**: Calculate for one root, then efficiently adjust for all others
2. **TWO-PASS ALGORITHM**: First pass gathers subtree info, second pass distributes results
3. **SUBTREE SIZE TRACKING**: Essential for efficient distance re-calculation
4. **LINEAR COMPLEXITY**: O(n) solution vs O(n²) naive approach
5. **TREE CENTER CONCEPT**: Node(s) minimizing total distance sum

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n²) time, O(n) space
   - Run BFS/DFS from each node to calculate distances
   - Simple but inefficient for large trees

2. **Tree DP with Re-rooting**: O(n) time, O(n) space
   - Optimal approach using two-pass technique
   - Industry standard for this type of problem

3. **Iterative Implementation**: O(n) time, O(n) space
   - Stack-based version of tree DP
   - Avoids recursion overhead

CORE RE-ROOTING ALGORITHM:
=========================
```python
def sumOfDistancesInTree(n, edges):
    # Build graph
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    subtree_size = [0] * n
    dist_sum = [0] * n
    
    # Pass 1: Calculate subtree sizes and distance sums for root 0
    def dfs1(node, parent):
        subtree_size[node] = 1
        for child in graph[node]:
            if child != parent:
                dfs1(child, node)
                subtree_size[node] += subtree_size[child]
                dist_sum[node] += dist_sum[child] + subtree_size[child]
    
    # Pass 2: Re-root to calculate answers for all nodes
    def dfs2(node, parent):
        for child in graph[node]:
            if child != parent:
                # Re-root from node to child
                dist_sum[child] = dist_sum[node] - subtree_size[child] + (n - subtree_size[child])
                dfs2(child, node)
    
    dfs1(0, -1)
    dfs2(0, -1)
    return dist_sum
```

RE-ROOTING MATHEMATICS:
======================
**First Pass (DFS1) - Rooted at node 0**:
```
subtree_size[node] = 1 + Σ(subtree_size[child])
dist_sum[node] = Σ(dist_sum[child] + subtree_size[child])
```

**Second Pass (DFS2) - Re-rooting**:
When moving root from `parent` to `child`:
```
new_dist_sum[child] = old_dist_sum[parent] - subtree_size[child] + (n - subtree_size[child])
```

**Intuition**: 
- Remove contribution of child's subtree (they get closer)
- Add contribution of non-child nodes (they get farther)

SUBTREE SIZE CALCULATION:
========================
**Purpose**: Count nodes in each subtree for distance calculations

**Calculation**:
```
subtree_size[node] = 1 + Σ(subtree_size[child] for each child)
```

**Usage in Distance Sum**:
- Each node in child's subtree is 1 unit farther when viewed from current node
- So we add `subtree_size[child]` to the distance sum

DISTANCE SUM PROPAGATION:
========================
**Upward Propagation (DFS1)**:
- Collect distance information from children
- Each child contributes: `dist_sum[child] + subtree_size[child]`
- The `+subtree_size[child]` accounts for increased distance

**Downward Propagation (DFS2)**:
- Distribute root's distance information to children
- Adjust for changed perspective when child becomes root

RE-ROOTING FORMULA DERIVATION:
=============================
**When re-rooting from `parent` to `child`**:

**Nodes in child's subtree**:
- Distance decreases by 1 for each of `subtree_size[child]` nodes
- Total change: `-subtree_size[child]`

**Nodes outside child's subtree**:
- Distance increases by 1 for each of `n - subtree_size[child]` nodes  
- Total change: `+(n - subtree_size[child])`

**Final formula**:
```
dist_sum[child] = dist_sum[parent] - subtree_size[child] + (n - subtree_size[child])
                = dist_sum[parent] + n - 2 * subtree_size[child]
```

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n) - two DFS passes, each visiting all nodes once
- **Space**: O(n) - graph storage + recursion stack + arrays
- **Improvement**: From O(n²) naive to O(n) optimal
- **Practical**: Very efficient even for large trees

TREE CENTER ANALYSIS:
====================
**Definition**: Node(s) minimizing sum of distances to all other nodes

**Properties**:
- Tree has at most 2 centers
- Centers are adjacent if there are 2
- Center minimizes maximum distance (tree radius)
- Often optimal for facility location problems

**Finding Centers**:
```python
dist_sums = sumOfDistancesInTree(n, edges)
min_sum = min(dist_sums)
centers = [i for i in range(n) if dist_sums[i] == min_sum]
```

APPLICATIONS:
============
- **Facility Location**: Optimal placement to minimize total travel distances
- **Network Design**: Central server placement in tree topologies
- **Logistics**: Distribution center optimization
- **Computational Biology**: Phylogenetic tree analysis
- **Social Networks**: Influence maximization in tree-structured networks

RELATED PROBLEMS:
================
- **Tree Diameter**: Longest path in tree (related to centers)
- **All Pairs Shortest Paths**: General graph version (more complex)
- **Centroid Decomposition**: Advanced tree decomposition technique
- **Tree Rerooting**: General framework for tree DP problems

VARIANTS:
========
**Weighted Distances**: Different weights for nodes or edges
**Limited Depth**: Only consider distances up to certain depth
**Multiple Facilities**: Place k facilities optimally
**Dynamic Trees**: Handle edge additions/deletions

EDGE CASES:
==========
- **Single Node**: Distance sum = 0
- **Linear Tree**: Center at middle node(s)
- **Star Graph**: Center at hub, all others equal
- **Balanced Binary Tree**: Root often optimal

OPTIMIZATION TECHNIQUES:
=======================
**Memory Optimization**: Iterative implementation for very large trees
**Cache Efficiency**: Optimize data structure layout
**Parallel Processing**: Independent subtree processing
**Approximation**: For very large graphs, approximate solutions

MATHEMATICAL PROPERTIES:
========================
- **Convexity**: Distance sum function is convex along tree paths
- **Uniqueness**: Optimal solution unique up to tree centers
- **Monotonicity**: Moving away from center increases distance sum
- **Symmetry**: Symmetric trees have symmetric distance distributions

This problem demonstrates the elegance and power of the re-rooting
technique, showing how clever algorithm design can reduce
complexity from quadratic to linear while solving complex
tree optimization problems efficiently.
"""
