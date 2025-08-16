"""
LeetCode 310: Minimum Height Trees
Difficulty: Medium
Category: Tree DP - Tree Center Finding

PROBLEM DESCRIPTION:
===================
A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you pick a node x as the root, the resulting tree has height h. Among all possible rooted trees, those with minimum height h are called minimum height trees (MHTs).

Return a list of all MHTs' root labels.

Example 1:
Input: n = 4, edges = [[1,0],[1,2],[1,3]]
Output: [1]
Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.

Example 2:
Input: n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
Output: [3,4]

Example 3:
Input: n = 1, edges = []
Output: [0]

Constraints:
- 1 <= n <= 2 * 10^4
- edges.length == n - 1
- 0 <= ai, bi < n
- ai != bi
- All the pairs (ai, bi) are distinct.
- The given input is guaranteed to form a tree.
"""

from collections import defaultdict, deque


def find_minimum_height_trees_brute_force(n, edges):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try each node as root and calculate height.
    
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
    
    def calculate_height(root):
        queue = deque([root])
        visited = {root}
        height = 0
        
        while queue:
            level_size = len(queue)
            height += 1
            
            for _ in range(level_size):
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return height - 1  # Height is levels - 1
    
    heights = []
    for i in range(n):
        heights.append(calculate_height(i))
    
    min_height = min(heights)
    return [i for i in range(n) if heights[i] == min_height]


def find_minimum_height_trees_leaf_removal(n, edges):
    """
    LEAF REMOVAL APPROACH (OPTIMAL):
    ===============================
    Repeatedly remove leaves until 1 or 2 nodes remain.
    
    Time Complexity: O(n) - each node removed exactly once
    Space Complexity: O(n) - graph storage
    """
    if n == 1:
        return [0]
    
    # Build adjacency list
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    
    # Find initial leaves (nodes with degree 1)
    leaves = deque()
    for i in range(n):
        if len(graph[i]) == 1:
            leaves.append(i)
    
    remaining_nodes = n
    
    # Remove leaves level by level
    while remaining_nodes > 2:
        leaves_count = len(leaves)
        remaining_nodes -= leaves_count
        
        # Remove current level of leaves
        for _ in range(leaves_count):
            leaf = leaves.popleft()
            
            # Remove leaf from its neighbor
            neighbor = graph[leaf].pop()  # Only one neighbor for leaf
            graph[neighbor].remove(leaf)
            
            # If neighbor becomes a leaf, add it to queue
            if len(graph[neighbor]) == 1:
                leaves.append(neighbor)
    
    # Return remaining nodes (tree centers)
    return list(range(n)) if remaining_nodes == n else [node for node in range(n) if graph[node]]


def find_minimum_height_trees_with_analysis(n, edges):
    """
    LEAF REMOVAL WITH DETAILED ANALYSIS:
    ===================================
    Track the removal process and analyze tree structure.
    
    Time Complexity: O(n) - optimal leaf removal
    Space Complexity: O(n) - tracking additional information
    """
    if n == 1:
        return [0], {"process": "Single node", "levels": 0}
    
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    
    # Track removal process
    removal_levels = []
    level = 0
    
    leaves = deque()
    for i in range(n):
        if len(graph[i]) == 1:
            leaves.append(i)
    
    remaining_nodes = n
    
    while remaining_nodes > 2:
        current_leaves = list(leaves)
        removal_levels.append({
            'level': level,
            'removed_leaves': current_leaves,
            'remaining_count': remaining_nodes - len(current_leaves)
        })
        
        leaves_count = len(leaves)
        remaining_nodes -= leaves_count
        level += 1
        
        for _ in range(leaves_count):
            leaf = leaves.popleft()
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            
            if len(graph[neighbor]) == 1:
                leaves.append(neighbor)
    
    # Find remaining centers
    centers = [node for node in range(n) if graph[node]]
    
    analysis = {
        'removal_levels': removal_levels,
        'total_levels': level,
        'final_centers': centers,
        'tree_radius': level
    }
    
    return centers, analysis


def find_minimum_height_trees_tree_dp(n, edges):
    """
    TREE DP APPROACH:
    ================
    Calculate height for each node using tree DP techniques.
    
    Time Complexity: O(n) - two DFS passes
    Space Complexity: O(n) - recursion + storage
    """
    if n == 1:
        return [0]
    
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    # First pass: calculate height down from each node
    down_height = [0] * n
    
    def dfs_down(node, parent):
        max_height = 0
        for child in graph[node]:
            if child != parent:
                dfs_down(child, node)
                max_height = max(max_height, 1 + down_height[child])
        down_height[node] = max_height
    
    # Second pass: calculate height up from each node
    up_height = [0] * n
    
    def dfs_up(node, parent, parent_up_height):
        up_height[node] = parent_up_height
        
        # Find two largest down heights from children
        child_heights = []
        for child in graph[node]:
            if child != parent:
                child_heights.append(down_height[child])
        
        child_heights.sort(reverse=True)
        if len(child_heights) < 2:
            child_heights.append(-1)
        
        for child in graph[node]:
            if child != parent:
                # Height going up through this node to child
                if down_height[child] == child_heights[0]:
                    max_other = child_heights[1]
                else:
                    max_other = child_heights[0]
                
                child_up_height = max(parent_up_height, max_other + 1) + 1
                dfs_up(child, node, child_up_height)
    
    # Calculate from arbitrary root
    dfs_down(0, -1)
    dfs_up(0, -1, 0)
    
    # Total height for each node as root
    heights = [max(down_height[i], up_height[i]) for i in range(n)]
    min_height = min(heights)
    
    return [i for i in range(n) if heights[i] == min_height]


def find_minimum_height_trees_diameter_based(n, edges):
    """
    DIAMETER-BASED APPROACH:
    =======================
    Find tree diameter and return center(s).
    
    Time Complexity: O(n) - two BFS calls
    Space Complexity: O(n) - graph and BFS storage
    """
    if n == 1:
        return [0]
    
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    def bfs_farthest(start):
        queue = deque([(start, 0)])
        visited = {start}
        farthest_node = start
        max_distance = 0
        distances = {start: 0}
        
        while queue:
            node, dist = queue.popleft()
            
            if dist > max_distance:
                max_distance = dist
                farthest_node = node
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        return farthest_node, max_distance, distances
    
    # Find one end of diameter
    end1, _, _ = bfs_farthest(0)
    
    # Find other end of diameter and path
    end2, diameter, distances_from_end1 = bfs_farthest(end1)
    
    # Reconstruct diameter path
    def reconstruct_path(start, end, distances):
        path = [end]
        current = end
        
        while current != start:
            for neighbor in graph[current]:
                if neighbor in distances and distances[neighbor] == distances[current] - 1:
                    path.append(neighbor)
                    current = neighbor
                    break
        
        return list(reversed(path))
    
    diameter_path = reconstruct_path(end1, end2, distances_from_end1)
    
    # Tree center is middle of diameter path
    path_length = len(diameter_path)
    if path_length % 2 == 1:
        # Odd length - single center
        return [diameter_path[path_length // 2]]
    else:
        # Even length - two centers
        mid = path_length // 2
        return [diameter_path[mid - 1], diameter_path[mid]]


def analyze_minimum_height_trees(n, edges):
    """
    COMPREHENSIVE MHT ANALYSIS:
    ==========================
    Analyze tree structure and minimum height tree properties.
    """
    if n == 1:
        print("Single node tree - root is node 0 with height 0")
        return [0]
    
    print(f"Minimum Height Trees Analysis for {n} nodes:")
    print(f"Number of edges: {len(edges)}")
    
    # Graph analysis
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)
    
    degrees = [len(graph[i]) for i in range(n)]
    print(f"Degree distribution: min={min(degrees)}, max={max(degrees)}, avg={sum(degrees)/n:.2f}")
    print(f"Leaves (degree 1): {degrees.count(1)}")
    
    # Different approaches
    brute_force_result = find_minimum_height_trees_brute_force(n, edges)
    leaf_removal_result = find_minimum_height_trees_leaf_removal(n, edges)
    tree_dp_result = find_minimum_height_trees_tree_dp(n, edges)
    diameter_based_result = find_minimum_height_trees_diameter_based(n, edges)
    
    print(f"\nResults Comparison:")
    print(f"Brute Force:     {brute_force_result}")
    print(f"Leaf Removal:    {leaf_removal_result}")
    print(f"Tree DP:         {tree_dp_result}")
    print(f"Diameter-based:  {diameter_based_result}")
    
    # Detailed analysis
    centers, analysis = find_minimum_height_trees_with_analysis(n, edges)
    
    print(f"\nDetailed Analysis:")
    print(f"Tree centers: {centers}")
    print(f"Tree radius: {analysis['tree_radius']}")
    print(f"Removal levels: {len(analysis['removal_levels'])}")
    
    for level_info in analysis['removal_levels']:
        print(f"  Level {level_info['level']}: Removed {level_info['removed_leaves']}, {level_info['remaining_count']} remaining")
    
    # Calculate actual heights
    def calculate_all_heights():
        heights = []
        for root in range(n):
            queue = deque([root])
            visited = {root}
            height = 0
            
            while queue:
                level_size = len(queue)
                if level_size == 0:
                    break
                height += 1
                
                for _ in range(level_size):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            heights.append(height - 1)
        return heights
    
    all_heights = calculate_all_heights()
    min_height = min(all_heights)
    
    print(f"\nHeight Analysis:")
    print(f"All heights: {all_heights}")
    print(f"Minimum height: {min_height}")
    print(f"Nodes achieving minimum height: {[i for i in range(n) if all_heights[i] == min_height]}")
    
    return centers


def minimum_height_trees_variants():
    """
    MINIMUM HEIGHT TREES VARIANTS:
    ==============================
    Different scenarios and modifications.
    """
    
    def find_k_minimum_height_trees(n, edges, k):
        """Find k nodes with smallest tree heights"""
        if n == 1:
            return [0]
        
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        def calculate_height(root):
            queue = deque([root])
            visited = {root}
            height = 0
            
            while queue:
                level_size = len(queue)
                if level_size == 0:
                    break
                height += 1
                
                for _ in range(level_size):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            return height - 1
        
        heights = [(calculate_height(i), i) for i in range(n)]
        heights.sort()
        
        return [node for _, node in heights[:k]]
    
    def find_minimum_weight_trees(n, edges, weights):
        """Find roots minimizing weighted tree height"""
        if n == 1:
            return [0]
        
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        def calculate_weighted_height(root):
            queue = deque([(root, 0)])
            visited = {root}
            max_weighted_depth = 0
            
            while queue:
                node, depth = queue.popleft()
                weighted_depth = depth * weights[node]
                max_weighted_depth = max(max_weighted_depth, weighted_depth)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            
            return max_weighted_depth
        
        weighted_heights = []
        for i in range(n):
            weighted_heights.append(calculate_weighted_height(i))
        
        min_weighted_height = min(weighted_heights)
        return [i for i in range(n) if weighted_heights[i] == min_weighted_height]
    
    def count_trees_with_height(n, edges, target_height):
        """Count number of roots giving specific height"""
        if n == 1:
            return 1 if target_height == 0 else 0
        
        graph = defaultdict(list)
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        
        def calculate_height(root):
            queue = deque([root])
            visited = {root}
            height = 0
            
            while queue:
                level_size = len(queue)
                if level_size == 0:
                    break
                height += 1
                
                for _ in range(level_size):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            
            return height - 1
        
        count = 0
        for i in range(n):
            if calculate_height(i) == target_height:
                count += 1
        
        return count
    
    # Test with sample trees
    test_cases = [
        (4, [[1,0],[1,2],[1,3]]),
        (6, [[3,0],[3,1],[3,2],[3,4],[5,4]]),
        (5, [[0,1],[0,2],[1,3],[1,4]])
    ]
    
    print("Minimum Height Trees Variants:")
    print("=" * 50)
    
    for i, (n, edges) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}: n={n}")
        
        basic_mht = find_minimum_height_trees_leaf_removal(n, edges)
        print(f"Basic MHT: {basic_mht}")
        
        k_smallest = find_k_minimum_height_trees(n, edges, 3)
        print(f"3 smallest heights: {k_smallest}")
        
        weights = [1, 2, 1, 1, 1, 1][:n]
        weighted_mht = find_minimum_weight_trees(n, edges, weights)
        print(f"Weighted MHT: {weighted_mht}")
        
        count_height_1 = count_trees_with_height(n, edges, 1)
        print(f"Trees with height 1: {count_height_1}")


# Test cases
def test_minimum_height_trees():
    """Test all implementations with various tree configurations"""
    
    test_cases = [
        (4, [[1,0],[1,2],[1,3]], [1]),
        (6, [[3,0],[3,1],[3,2],[3,4],[5,4]], [3,4]),
        (1, [], [0]),
        (2, [[0,1]], [0,1]),
        (3, [[0,1],[1,2]], [1]),
        (5, [[0,1],[0,2],[1,3],[1,4]], [1]),
        (7, [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6]], [1,2])
    ]
    
    print("Testing Minimum Height Trees Solutions:")
    print("=" * 70)
    
    for i, (n, edges, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"n = {n}, edges = {edges}")
        print(f"Expected: {expected}")
        
        # Test all approaches
        brute_force = find_minimum_height_trees_brute_force(n, edges)
        leaf_removal = find_minimum_height_trees_leaf_removal(n, edges)
        tree_dp = find_minimum_height_trees_tree_dp(n, edges)
        diameter_based = find_minimum_height_trees_diameter_based(n, edges)
        
        print(f"Brute Force:      {sorted(brute_force)} {'✓' if sorted(brute_force) == sorted(expected) else '✗'}")
        print(f"Leaf Removal:     {sorted(leaf_removal)} {'✓' if sorted(leaf_removal) == sorted(expected) else '✗'}")
        print(f"Tree DP:          {sorted(tree_dp)} {'✓' if sorted(tree_dp) == sorted(expected) else '✗'}")
        print(f"Diameter-based:   {sorted(diameter_based)} {'✓' if sorted(diameter_based) == sorted(expected) else '✗'}")
    
    # Comprehensive analysis example
    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE ANALYSIS EXAMPLE:")
    print("-" * 40)
    analyze_minimum_height_trees(6, [[3,0],[3,1],[3,2],[3,4],[5,4]])
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    minimum_height_trees_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. TREE CENTERS: MHT roots are tree centers (at most 2)")
    print("2. LEAF REMOVAL: Optimal O(n) algorithm using BFS-like approach")
    print("3. DIAMETER RELATIONSHIP: Centers are middle of longest path")
    print("4. UNIQUE SOLUTION: Tree has at most 2 centers")
    print("5. RADIUS MINIMIZATION: Centers minimize maximum distance to any node")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Network Design: Optimal root placement for minimum latency")
    print("• Facility Location: Central facility placement")
    print("• Tree Algorithms: Optimal root selection for tree processing")
    print("• Graph Theory: Tree center and diameter calculations")
    print("• Distributed Systems: Coordinator placement optimization")


if __name__ == "__main__":
    test_minimum_height_trees()


"""
MINIMUM HEIGHT TREES - TREE CENTER FINDING OPTIMIZATION:
========================================================

This problem demonstrates the elegant concept of tree centers:
- Finding roots that minimize tree height (radius)
- Multiple optimal algorithms with different perspectives
- Connection between tree centers and diameter
- Practical applications in network and facility optimization

KEY INSIGHTS:
============
1. **TREE CENTERS**: MHT roots are tree centers (at most 2 exist)
2. **LEAF REMOVAL**: Optimal O(n) algorithm peeling leaves level by level
3. **DIAMETER CONNECTION**: Centers are at the middle of the longest path
4. **UNIQUENESS**: Every tree has exactly 1 or 2 centers
5. **RADIUS MINIMIZATION**: Centers minimize maximum distance to any node

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n²) time, O(n) space
   - Calculate height for each possible root
   - Simple but inefficient for large trees

2. **Leaf Removal (Optimal)**: O(n) time, O(n) space
   - Repeatedly remove leaves until 1-2 nodes remain
   - Most elegant and efficient approach

3. **Tree DP**: O(n) time, O(n) space
   - Two-pass algorithm calculating up/down heights
   - Demonstrates re-rooting technique

4. **Diameter-based**: O(n) time, O(n) space
   - Find diameter endpoints, return middle node(s)
   - Connects to tree diameter theory

CORE LEAF REMOVAL ALGORITHM:
============================
```python
def findMinHeightTrees(n, edges):
    if n == 1:
        return [0]
    
    # Build graph
    graph = defaultdict(set)
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    
    # Find initial leaves
    leaves = deque()
    for i in range(n):
        if len(graph[i]) == 1:
            leaves.append(i)
    
    remaining = n
    
    # Remove leaves level by level
    while remaining > 2:
        leaf_count = len(leaves)
        remaining -= leaf_count
        
        for _ in range(leaf_count):
            leaf = leaves.popleft()
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            
            if len(graph[neighbor]) == 1:
                leaves.append(neighbor)
    
    return list(graph.keys()) if remaining == n else [i for i in range(n) if graph[i]]
```

WHY LEAF REMOVAL WORKS:
======================
**Tree Center Properties**:
- Tree center minimizes maximum distance to any node
- If we remove leaves, the center doesn't change
- Eventually only the center(s) remain

**Proof Sketch**:
- Optimal root cannot be a leaf (internal node always better)
- Removing leaves preserves relative optimality
- Process terminates when only 1-2 nodes remain
- These are exactly the tree centers

TREE CENTER MATHEMATICS:
=======================
**Center Count**: Every tree has exactly 1 or 2 centers
- If diameter is even: 2 centers (adjacent)
- If diameter is odd: 1 center

**Center Location**: 
- Centers are at the middle of the diameter path
- Distance from center to any node ≤ ⌊diameter/2⌋

**Optimality**: 
- Tree rooted at center has minimum height
- Height = ⌊diameter/2⌋

DIAMETER-CENTER CONNECTION:
==========================
**Diameter**: Longest path in the tree

**Finding Diameter**:
1. BFS from arbitrary node to find farthest node
2. BFS from that node to find actual diameter endpoints
3. Centers are middle node(s) of diameter path

**Mathematical Relationship**:
```
tree_radius = ⌊diameter/2⌋
center_count = 2 - (diameter % 2)
```

TREE DP APPROACH:
================
**Two-Pass Algorithm**:
1. **Down Pass**: Calculate maximum depth in each subtree
2. **Up Pass**: Calculate maximum depth going up through parent

**Height Calculation**:
```
height[node] = max(down_height[node], up_height[node])
```

**Re-rooting Formula**:
When moving from parent to child, update up_height based on:
- Height through parent's parent
- Maximum height from parent's other children

COMPLEXITY ANALYSIS:
===================
- **Leaf Removal**: O(n) time, O(n) space - optimal
- **Tree DP**: O(n) time, O(n) space - demonstrates advanced techniques
- **Diameter-based**: O(n) time, O(n) space - connects to diameter theory
- **Brute Force**: O(n²) time, O(n) space - simple but inefficient

APPLICATIONS:
============
- **Network Design**: Optimal root server placement for minimum latency
- **Facility Location**: Central distribution center placement
- **Database Systems**: B-tree root optimization
- **Distributed Computing**: Coordinator node selection
- **Social Networks**: Influence maximization starting points

RELATED PROBLEMS:
================
- **Tree Diameter**: Finding longest path in tree
- **Tree Centroid**: Different center concept for divide-and-conquer
- **All Pairs Shortest Paths**: General graph center finding
- **Facility Location**: k-center and k-median problems

VARIANTS:
========
**Weighted Trees**: Different edge or node weights
**k-Centers**: Find k optimal roots for forest
**Dynamic Trees**: Handle edge additions/deletions
**Constrained Centers**: Additional constraints on valid roots

EDGE CASES:
==========
- **Single Node**: Only possible root with height 0
- **Two Nodes**: Both nodes are centers with height 0
- **Linear Tree**: Middle node(s) are centers
- **Star Graph**: Central node is unique center

OPTIMIZATION TECHNIQUES:
=======================
**Memory Optimization**: Use sets for efficient neighbor removal
**Early Termination**: Stop when ≤2 nodes remain
**Parallel Processing**: Independent subtree processing possible
**Cache Efficiency**: Optimize data structure layout

MATHEMATICAL PROPERTIES:
========================
- **Uniqueness**: At most 2 centers exist
- **Adjacency**: If 2 centers exist, they're adjacent
- **Optimality**: Centers minimize tree radius
- **Invariance**: Centers don't change under leaf removal

**Proof of at most 2 centers**:
- Suppose 3 centers exist: a, b, c
- Path from a to c must go through b
- Then distance from b to some node < distance from a or c
- Contradiction with all being centers

This problem beautifully demonstrates the connection between
different tree concepts (diameter, center, radius) and shows
how elegant algorithms can solve complex optimization problems
efficiently using fundamental graph properties.
"""
