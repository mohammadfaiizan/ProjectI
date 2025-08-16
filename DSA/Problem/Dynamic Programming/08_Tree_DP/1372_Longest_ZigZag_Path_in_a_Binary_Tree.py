"""
LeetCode 1372: Longest ZigZag Path in a Binary Tree
Difficulty: Medium
Category: Tree DP - Path Pattern Optimization

PROBLEM DESCRIPTION:
===================
You are given the root of a binary tree.

A ZigZag path for a binary tree is defined as follow:
- Choose any node in the binary tree and a direction (right or left).
- If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
- Change the direction from right to left or from left to right.
- Repeat the second and third steps until you can't move in the tree.

Zigzag length is defined as the number of nodes visited - 1. (A single node has a length of 0).

Return the length of the longest ZigZag path contained in that tree.

Example 1:
Input: root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1]
Output: 3
Explanation: Longest ZigZag path in blue nodes (right -> left -> right).

Example 2:
Input: root = [1,1,1,null,1,null,null,1,1,null,1]
Output: 4
Explanation: Longest ZigZag path in blue nodes (left -> right -> left -> right).

Example 3:
Input: root = [1]
Output: 0

Constraints:
- The number of nodes in the tree is in the range [1, 5 * 10^4].
- 1 <= Node.val <= 100
"""

# TreeNode definition for reference
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def longest_zigzag_tree_dp(root):
    """
    TREE DP APPROACH:
    ================
    For each node, track longest zigzag ending at that node from left and right.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    max_zigzag = 0
    
    def dfs(node):
        nonlocal max_zigzag
        
        if not node:
            return (0, 0)  # (left_zigzag, right_zigzag)
        
        left_result = dfs(node.left)
        right_result = dfs(node.right)
        
        # Zigzag ending at current node going left (from right child)
        left_zigzag = right_result[1] + 1 if node.right else 0
        
        # Zigzag ending at current node going right (from left child)
        right_zigzag = left_result[0] + 1 if node.left else 0
        
        # Update global maximum
        max_zigzag = max(max_zigzag, left_zigzag, right_zigzag)
        
        return (left_zigzag, right_zigzag)
    
    dfs(root)
    return max_zigzag


def longest_zigzag_with_path_tracking(root):
    """
    ZIGZAG WITH PATH TRACKING:
    =========================
    Track the actual longest zigzag path.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion depth + path storage
    """
    if not root:
        return 0, []
    
    max_zigzag = 0
    best_path = []
    
    def dfs(node, path=[]):
        nonlocal max_zigzag, best_path
        
        if not node:
            return (0, 0, [], [])  # (left_zigzag, right_zigzag, left_path, right_path)
        
        current_path = path + [node.val]
        
        left_result = dfs(node.left, current_path)
        right_result = dfs(node.right, current_path)
        
        # Zigzag going left from current node
        left_zigzag = right_result[1] + 1 if node.right else 0
        left_path = current_path + right_result[3][1:] if node.right and right_result[3] else current_path
        
        # Zigzag going right from current node
        right_zigzag = left_result[0] + 1 if node.left else 0
        right_path = current_path + left_result[2][1:] if node.left and left_result[2] else current_path
        
        # Update global maximum
        if left_zigzag > max_zigzag:
            max_zigzag = left_zigzag
            best_path = left_path.copy()
        
        if right_zigzag > max_zigzag:
            max_zigzag = right_zigzag
            best_path = right_path.copy()
        
        return (left_zigzag, right_zigzag, left_path, right_path)
    
    dfs(root)
    return max_zigzag, best_path


def longest_zigzag_all_paths(root):
    """
    FIND ALL ZIGZAG PATHS:
    =====================
    Find all possible zigzag paths and their lengths.
    
    Time Complexity: O(n^2) - potentially many paths
    Space Complexity: O(n) - path storage
    """
    if not root:
        return []
    
    all_zigzag_paths = []
    
    def dfs_from_node(node, direction, path, length):
        if not node:
            return
        
        current_path = path + [node.val]
        all_zigzag_paths.append((length, current_path.copy(), direction))
        
        # Continue zigzag pattern
        if direction == "left":
            # Next must go right
            if node.right:
                dfs_from_node(node.right, "right", current_path, length + 1)
        elif direction == "right":
            # Next must go left
            if node.left:
                dfs_from_node(node.left, "left", current_path, length + 1)
        else:  # direction == "start"
            # Can start in either direction
            if node.left:
                dfs_from_node(node.left, "left", current_path, length + 1)
            if node.right:
                dfs_from_node(node.right, "right", current_path, length + 1)
    
    def explore_all_starting_points(node):
        if not node:
            return
        
        # Try starting from this node
        dfs_from_node(node, "start", [], 0)
        
        # Recurse to children
        explore_all_starting_points(node.left)
        explore_all_starting_points(node.right)
    
    explore_all_starting_points(root)
    
    # Sort by length and return
    all_zigzag_paths.sort(reverse=True, key=lambda x: x[0])
    return all_zigzag_paths


def longest_zigzag_iterative(root):
    """
    ITERATIVE APPROACH:
    ==================
    Use stack to implement zigzag path finding.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - stack space
    """
    if not root:
        return 0
    
    # Stack: (node, direction, length)
    # direction: 0 = left, 1 = right, -1 = can go either way
    stack = [(root, -1, 0)]
    max_zigzag = 0
    
    while stack:
        node, direction, length = stack.pop()
        
        if not node:
            continue
        
        max_zigzag = max(max_zigzag, length)
        
        # If we came from left, next must go right
        if direction == 0:
            if node.right:
                stack.append((node.right, 1, length + 1))
            if node.left:
                stack.append((node.left, 0, 1))
        
        # If we came from right, next must go left
        elif direction == 1:
            if node.left:
                stack.append((node.left, 0, length + 1))
            if node.right:
                stack.append((node.right, 1, 1))
        
        # Starting node - can go either direction
        else:
            if node.left:
                stack.append((node.left, 0, 1))
            if node.right:
                stack.append((node.right, 1, 1))
    
    return max_zigzag


def longest_zigzag_with_state_tracking(root):
    """
    DETAILED STATE TRACKING:
    =======================
    Track zigzag states with detailed analysis.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    max_zigzag = 0
    state_info = {}
    
    def dfs(node, node_id="root"):
        nonlocal max_zigzag
        
        if not node:
            return (0, 0)
        
        left_id = f"{node_id}_L"
        right_id = f"{node_id}_R"
        
        left_result = dfs(node.left, left_id)
        right_result = dfs(node.right, right_id)
        
        # Calculate zigzag lengths
        left_zigzag = right_result[1] + 1 if node.right else 0
        right_zigzag = left_result[0] + 1 if node.left else 0
        
        # Store state information
        state_info[node_id] = {
            'val': node.val,
            'left_zigzag': left_zigzag,
            'right_zigzag': right_zigzag,
            'max_local': max(left_zigzag, right_zigzag),
            'contributes_to_left': left_result[0] if node.left else 0,
            'contributes_to_right': right_result[1] if node.right else 0
        }
        
        max_zigzag = max(max_zigzag, left_zigzag, right_zigzag)
        
        return (left_zigzag, right_zigzag)
    
    dfs(root)
    
    return max_zigzag, state_info


def longest_zigzag_analysis(root):
    """
    COMPREHENSIVE ZIGZAG ANALYSIS:
    ==============================
    Analyze zigzag patterns and tree structure.
    """
    if not root:
        print("Empty tree - no zigzag paths possible!")
        return 0
    
    # Tree structure analysis
    def analyze_tree_structure(node, depth=0):
        if not node:
            return {
                'nodes': 0, 'leaves': 0, 'max_depth': depth,
                'left_only': 0, 'right_only': 0, 'both_children': 0
            }
        
        left_info = analyze_tree_structure(node.left, depth + 1)
        right_info = analyze_tree_structure(node.right, depth + 1)
        
        node_type = ('left_only' if node.left and not node.right else
                    'right_only' if node.right and not node.left else
                    'both_children' if node.left and node.right else
                    'leaf')
        
        return {
            'nodes': 1 + left_info['nodes'] + right_info['nodes'],
            'leaves': (1 if node_type == 'leaf' else 0) + left_info['leaves'] + right_info['leaves'],
            'max_depth': max(left_info['max_depth'], right_info['max_depth']),
            'left_only': (1 if node_type == 'left_only' else 0) + left_info['left_only'] + right_info['left_only'],
            'right_only': (1 if node_type == 'right_only' else 0) + left_info['right_only'] + right_info['right_only'],
            'both_children': (1 if node_type == 'both_children' else 0) + left_info['both_children'] + right_info['both_children']
        }
    
    tree_info = analyze_tree_structure(root)
    
    print(f"Tree Structure Analysis:")
    print(f"  Total nodes: {tree_info['nodes']}")
    print(f"  Leaf nodes: {tree_info['leaves']}")
    print(f"  Tree height: {tree_info['max_depth']}")
    print(f"  Left-only children: {tree_info['left_only']}")
    print(f"  Right-only children: {tree_info['right_only']}")
    print(f"  Both children: {tree_info['both_children']}")
    
    # Zigzag analysis
    max_length = longest_zigzag_tree_dp(root)
    max_with_path, best_path = longest_zigzag_with_path_tracking(root)
    max_with_state, state_info = longest_zigzag_with_state_tracking(root)
    
    print(f"\nZigzag Path Analysis:")
    print(f"  Longest zigzag length: {max_length}")
    print(f"  Best path: {best_path}")
    print(f"  Efficiency: {max_length/tree_info['max_depth']:.2f} (length/height ratio)")
    
    # Pattern analysis
    all_paths = longest_zigzag_all_paths(root)
    if all_paths:
        print(f"\nTop 5 Zigzag Paths:")
        for i, (length, path, direction) in enumerate(all_paths[:5]):
            print(f"  {i+1}. Length {length}: {path} (started going {direction})")
    
    # State analysis
    print(f"\nNode State Analysis (first 5 nodes):")
    for i, (node_id, info) in enumerate(list(state_info.items())[:5]):
        print(f"  {node_id}: val={info['val']}, left_zigzag={info['left_zigzag']}, right_zigzag={info['right_zigzag']}")
    
    return max_length


def longest_zigzag_variants():
    """
    ZIGZAG PATH VARIANTS:
    ====================
    Different scenarios and modifications.
    """
    
    def longest_zigzag_with_min_length(root, min_length):
        """Find longest zigzag path with at least min_length"""
        if not root:
            return 0
        
        max_zigzag = 0
        
        def dfs(node):
            nonlocal max_zigzag
            
            if not node:
                return (0, 0)
            
            left_result = dfs(node.left)
            right_result = dfs(node.right)
            
            left_zigzag = right_result[1] + 1 if node.right else 0
            right_zigzag = left_result[0] + 1 if node.left else 0
            
            if left_zigzag >= min_length:
                max_zigzag = max(max_zigzag, left_zigzag)
            if right_zigzag >= min_length:
                max_zigzag = max(max_zigzag, right_zigzag)
            
            return (left_zigzag, right_zigzag)
        
        dfs(root)
        return max_zigzag if max_zigzag >= min_length else -1
    
    def count_zigzag_paths_of_length(root, target_length):
        """Count number of zigzag paths of specific length"""
        if not root:
            return 0
        
        count = 0
        
        def dfs_count(node, direction, current_length):
            nonlocal count
            
            if not node:
                return
            
            if current_length == target_length:
                count += 1
                return
            
            # Continue zigzag pattern
            if direction == "left":
                if node.right:
                    dfs_count(node.right, "right", current_length + 1)
            elif direction == "right":
                if node.left:
                    dfs_count(node.left, "left", current_length + 1)
            else:  # starting
                if node.left:
                    dfs_count(node.left, "left", current_length + 1)
                if node.right:
                    dfs_count(node.right, "right", current_length + 1)
        
        def explore_all_starts(node):
            if not node:
                return
            
            dfs_count(node, "start", 0)
            explore_all_starts(node.left)
            explore_all_starts(node.right)
        
        explore_all_starts(root)
        return count
    
    def longest_weighted_zigzag(root, weights):
        """Zigzag with weighted nodes"""
        if not root:
            return 0
        
        max_weight = 0
        
        def dfs(node, node_index=0):
            nonlocal max_weight
            
            if not node:
                return (0, 0)  # (left_weight, right_weight)
            
            left_result = dfs(node.left, node_index * 2 + 1)
            right_result = dfs(node.right, node_index * 2 + 2)
            
            node_weight = weights[node_index] if node_index < len(weights) else 1
            
            left_zigzag = right_result[1] + node_weight if node.right else 0
            right_zigzag = left_result[0] + node_weight if node.left else 0
            
            max_weight = max(max_weight, left_zigzag, right_zigzag)
            
            return (left_zigzag, right_zigzag)
        
        dfs(root)
        return max_weight
    
    # Test with sample tree
    def create_sample_tree():
        """Create sample tree for testing"""
        root = TreeNode(1)
        root.right = TreeNode(1)
        root.right.left = TreeNode(1)
        root.right.right = TreeNode(1)
        root.right.right.left = TreeNode(1)
        root.right.right.left.right = TreeNode(1)
        return root
    
    sample_tree = create_sample_tree()
    
    print("Zigzag Path Variants:")
    print("=" * 50)
    
    basic_zigzag = longest_zigzag_tree_dp(sample_tree)
    print(f"Basic longest zigzag: {basic_zigzag}")
    
    min_length_zigzag = longest_zigzag_with_min_length(sample_tree, 2)
    print(f"Longest zigzag with min length 2: {min_length_zigzag}")
    
    count_specific = count_zigzag_paths_of_length(sample_tree, 2)
    print(f"Number of zigzag paths of length 2: {count_specific}")
    
    weights = [1, 2, 3, 4, 5, 6]
    weighted_zigzag = longest_weighted_zigzag(sample_tree, weights)
    print(f"Longest weighted zigzag: {weighted_zigzag}")


# Test cases
def test_longest_zigzag():
    """Test all implementations with various tree configurations"""
    
    def create_tree_from_list(values):
        """Helper to create tree from level-order list"""
        if not values:
            return None
        
        root = TreeNode(values[0])
        queue = [root]
        i = 1
        
        while queue and i < len(values):
            node = queue.pop(0)
            
            if i < len(values) and values[i] is not None:
                node.left = TreeNode(values[i])
                queue.append(node.left)
            i += 1
            
            if i < len(values) and values[i] is not None:
                node.right = TreeNode(values[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    test_cases = [
        ([1, None, 1, 1, 1, None, None, 1, 1, None, 1, None, None, None, 1], 3),
        ([1, 1, 1, None, 1, None, None, 1, 1, None, 1], 4),
        ([1], 0),
        ([1, 1], 1),
        ([1, None, 1], 1),
        ([1, 1, None, 1, None, 1], 2),
        ([1, 1, 1, 1, 1, 1, 1], 2)
    ]
    
    print("Testing Longest ZigZag Path in Binary Tree Solutions:")
    print("=" * 70)
    
    for i, (tree_list, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Tree: {tree_list}")
        print(f"Expected: {expected}")
        
        root = create_tree_from_list(tree_list)
        
        tree_dp = longest_zigzag_tree_dp(root)
        iterative = longest_zigzag_iterative(root)
        
        print(f"Tree DP:          {tree_dp:>4} {'✓' if tree_dp == expected else '✗'}")
        print(f"Iterative:        {iterative:>4} {'✓' if iterative == expected else '✗'}")
        
        # Show path for small cases
        if len([x for x in tree_list if x is not None]) <= 8:
            max_with_path, best_path = longest_zigzag_with_path_tracking(root)
            print(f"Best zigzag path: {best_path} (length: {max_with_path})")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    sample_tree = create_tree_from_list([1, None, 1, 1, 1, None, None, 1, 1, None, 1])
    longest_zigzag_analysis(sample_tree)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    longest_zigzag_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. ALTERNATING PATTERN: ZigZag requires alternating left-right movements")
    print("2. STATE TRACKING: Each node tracks zigzag lengths in both directions")
    print("3. DIRECTION DEPENDENCY: Current direction determines next valid move")
    print("4. GLOBAL MAXIMUM: Track maximum across all possible starting points")
    print("5. OPTIMAL SUBSTRUCTURE: Optimal zigzag contains optimal sub-zigzags")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Path Optimization: Finding optimal alternating patterns in networks")
    print("• Game Theory: Optimal movement strategies in constrained environments")
    print("• Robot Navigation: Optimal zigzag paths for coverage or evasion")
    print("• Algorithm Design: Pattern matching in tree structures")
    print("• Network Routing: Alternating path selection for load balancing")


if __name__ == "__main__":
    test_longest_zigzag()


"""
LONGEST ZIGZAG PATH IN BINARY TREE - PATTERN OPTIMIZATION:
==========================================================

This problem demonstrates advanced Tree DP for pattern-constrained paths:
- Alternating direction constraints (left-right-left-right)
- State-dependent transitions (direction affects next valid moves)
- Global optimization across all possible starting points
- Pattern validation through state tracking

KEY INSIGHTS:
============
1. **ALTERNATING PATTERN**: ZigZag requires strict left-right alternation
2. **DIRECTIONAL STATE**: Each node tracks zigzag lengths for both directions
3. **STATE DEPENDENCY**: Current direction determines valid next moves
4. **GLOBAL TRACKING**: Must consider all possible starting points
5. **OPTIMAL SUBSTRUCTURE**: Optimal zigzag paths contain optimal sub-zigzags

ALGORITHM APPROACHES:
====================

1. **Tree DP (Optimal)**: O(n) time, O(h) space
   - Track left and right zigzag lengths at each node
   - Most efficient approach

2. **Iterative DFS**: O(n) time, O(h) space
   - Stack-based traversal with state tracking
   - Avoids recursion overhead

3. **All Paths Enumeration**: O(n²) time, O(n) space
   - Find all possible zigzag paths
   - Good for analysis but less efficient

CORE TREE DP ALGORITHM:
======================
```python
def longestZigZag(root):
    max_zigzag = 0
    
    def dfs(node):
        nonlocal max_zigzag
        
        if not node:
            return (0, 0)  # (left_zigzag, right_zigzag)
        
        left_result = dfs(node.left)
        right_result = dfs(node.right)
        
        # ZigZag ending at current node going left (from right child)
        left_zigzag = right_result[1] + 1 if node.right else 0
        
        # ZigZag ending at current node going right (from left child)  
        right_zigzag = left_result[0] + 1 if node.left else 0
        
        max_zigzag = max(max_zigzag, left_zigzag, right_zigzag)
        
        return (left_zigzag, right_zigzag)
    
    dfs(root)
    return max_zigzag
```

STATE TRANSITION LOGIC:
======================
**State Definition**:
- `left_zigzag[node]`: Longest zigzag ending at node, last move was left
- `right_zigzag[node]`: Longest zigzag ending at node, last move was right

**Transition Rules**:
```
left_zigzag[node] = right_zigzag[right_child] + 1   (if right child exists)
right_zigzag[node] = left_zigzag[left_child] + 1    (if left child exists)
```

**Pattern Enforcement**: 
- To go left at current node, previous move must have been right
- To go right at current node, previous move must have been left

DIRECTION CONSTRAINT ANALYSIS:
=============================
**ZigZag Pattern Requirements**:
1. **Alternation**: No two consecutive moves in same direction
2. **Continuity**: Each move extends from previous position
3. **Validity**: Can only move to existing children

**State Propagation**:
- Left zigzag can only extend from right zigzag of opposite child
- Right zigzag can only extend from left zigzag of opposite child
- No direct extension from same-direction parent zigzag

PATH STARTING POINT OPTIMIZATION:
=================================
**Global Maximum Strategy**:
- Consider every node as potential starting point
- Each node contributes its best zigzag in either direction
- Global maximum = max over all nodes of their best local zigzag

**Why This Works**:
- Optimal zigzag must end at some node
- By considering all ending points, we find global optimum
- Tree DP automatically considers all starting points through recursion

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n) - each node visited exactly once
- **Space**: O(h) - recursion stack depth
- **Optimal**: Cannot improve (must examine all nodes for pattern validation)
- **Practical**: Very efficient with minimal overhead

PATTERN VALIDATION FRAMEWORK:
============================
**ZigZag Validation Rules**:
1. **Length ≥ 0**: Single node has zigzag length 0
2. **Alternation**: Direction must change between consecutive moves
3. **Connectivity**: Each move must go to existing child
4. **Maximality**: Cannot be extended while maintaining pattern

**Implementation Validation**:
- State transitions enforce alternation automatically
- Child existence checks ensure connectivity
- Global tracking finds maximum length

APPLICATIONS:
============
- **Path Optimization**: Optimal alternating patterns in networks
- **Robot Navigation**: Zigzag movement for coverage or obstacle avoidance
- **Game Theory**: Optimal strategies with movement constraints
- **Pattern Recognition**: Finding specific patterns in tree data
- **Network Routing**: Alternating path selection for load distribution

RELATED PROBLEMS:
================
- **Binary Tree Maximum Path Sum**: Similar tree traversal with different constraints
- **House Robber III**: Constraint-based optimization in trees
- **Binary Tree Diameter**: Path length optimization (but different constraints)
- **Longest Path in Graph**: General path optimization (trees are special case)

VARIANTS:
========
**Length Constraints**: Minimum or maximum zigzag length requirements
**Weighted ZigZag**: Different weights for nodes or edges
**K-Pattern**: Generalize to k-step repeating patterns
**Multiple Patterns**: Find multiple non-overlapping zigzag paths

EDGE CASES:
==========
- **Single Node**: ZigZag length = 0 (no movement possible)
- **Linear Tree**: Maximum length = height - 1
- **Perfect Binary Tree**: Multiple optimal zigzag paths possible
- **Unbalanced Tree**: Zigzag length limited by structure

OPTIMIZATION TECHNIQUES:
=======================
**State Compression**: Only need current and child states
**Early Termination**: Prune when no improvement possible
**Path Reconstruction**: Track actual paths for analysis
**Memory Optimization**: Iterative implementation for very deep trees

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Global optimum contains local optima
- **No Overlapping Subproblems**: Tree structure prevents this
- **Monotonicity**: Adding valid extensions never decreases length
- **Pattern Preservation**: Optimal extensions maintain zigzag property

This problem elegantly demonstrates how Tree DP can handle
complex pattern constraints efficiently, showing how
state-dependent transitions can enforce specific movement
patterns while maintaining optimal solution guarantees.
"""
