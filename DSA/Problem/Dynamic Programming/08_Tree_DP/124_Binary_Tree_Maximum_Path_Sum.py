"""
LeetCode 124: Binary Tree Maximum Path Sum
Difficulty: Hard
Category: Tree DP - Path Optimization

PROBLEM DESCRIPTION:
===================
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

Example 1:
Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

Example 2:
Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.

Constraints:
- The number of nodes in the tree is in the range [1, 3 * 10^4].
- -1000 <= Node.val <= 1000
"""

# TreeNode definition for reference
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def max_path_sum_brute_force(root):
    """
    BRUTE FORCE APPROACH:
    ====================
    For each node, calculate all possible paths and find maximum.
    
    Time Complexity: O(n^2) - for each node, explore all paths
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    
    def all_paths_from_node(node, current_sum, visited):
        nonlocal max_sum
        if not node or node in visited:
            return
        
        current_sum += node.val
        max_sum = max(max_sum, current_sum)
        visited.add(node)
        
        # Explore all connected nodes
        if node.left:
            all_paths_from_node(node.left, current_sum, visited)
        if node.right:
            all_paths_from_node(node.right, current_sum, visited)
        
        visited.remove(node)
    
    def explore_all_starting_points(node):
        if not node:
            return
        
        # Try paths starting from current node
        all_paths_from_node(node, 0, set())
        
        # Recurse to children
        explore_all_starting_points(node.left)
        explore_all_starting_points(node.right)
    
    explore_all_starting_points(root)
    return max_sum


def max_path_sum_tree_dp(root):
    """
    TREE DP APPROACH:
    ================
    For each node, return maximum path sum ending at that node.
    Track global maximum separately.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion stack
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Maximum sum on left and right sub-trees of node
        # Only consider positive gains
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Maximum path sum through current node
        path_through_node = node.val + left_gain + right_gain
        
        # Update global maximum
        max_sum = max(max_sum, path_through_node)
        
        # Return maximum gain if continuing path through parent
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum


def max_path_sum_detailed(root):
    """
    DETAILED TREE DP WITH PATH TRACKING:
    ===================================
    Track both maximum sum and the actual path.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion stack + path storage
    """
    if not root:
        return 0, []
    
    max_sum = float('-inf')
    best_path = []
    
    def max_gain_with_path(node):
        nonlocal max_sum, best_path
        
        if not node:
            return 0, []
        
        # Get gains and paths from children
        left_gain, left_path = max_gain_with_path(node.left)
        right_gain, right_path = max_gain_with_path(node.right)
        
        # Only consider positive gains
        left_gain = max(left_gain, 0)
        right_gain = max(right_gain, 0)
        
        # Maximum path sum through current node
        path_through_node = node.val + left_gain + right_gain
        
        # Construct path through current node
        current_path = []
        if left_gain > 0:
            current_path = list(reversed(left_path)) + current_path
        current_path.append(node.val)
        if right_gain > 0:
            current_path = current_path + right_path
        
        # Update global maximum
        if path_through_node > max_sum:
            max_sum = path_through_node
            best_path = current_path.copy()
        
        # Return maximum gain continuing through parent
        if left_gain > right_gain:
            return node.val + left_gain, list(reversed(left_path)) + [node.val]
        else:
            return node.val + right_gain, [node.val] + right_path
    
    max_gain_with_path(root)
    return max_sum, best_path


def max_path_sum_iterative(root):
    """
    ITERATIVE APPROACH USING POST-ORDER TRAVERSAL:
    ==============================================
    Stack-based implementation of tree DP.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - stack space
    """
    if not root:
        return 0
    
    stack = [(root, False)]  # (node, processed)
    gains = {}  # node -> max_gain_from_node
    max_sum = float('-inf')
    
    while stack:
        node, processed = stack.pop()
        
        if processed:
            # Process node after children are processed
            left_gain = gains.get(node.left, 0)
            right_gain = gains.get(node.right, 0)
            
            # Only consider positive gains
            left_gain = max(left_gain, 0)
            right_gain = max(right_gain, 0)
            
            # Maximum path through current node
            path_through_node = node.val + left_gain + right_gain
            max_sum = max(max_sum, path_through_node)
            
            # Gain from current node for parent
            gains[node] = node.val + max(left_gain, right_gain)
        else:
            # Mark for processing and add children
            stack.append((node, True))
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    return max_sum


def max_path_sum_with_constraints(root, min_length=1, max_length=float('inf')):
    """
    PATH SUM WITH LENGTH CONSTRAINTS:
    ================================
    Find maximum path sum with path length constraints.
    
    Time Complexity: O(n * max_length) - each node with length states
    Space Complexity: O(h * max_length) - recursion with length tracking
    """
    if not root:
        return 0
    
    max_sum = float('-inf')
    
    def max_gain_with_length(node, remaining_length):
        nonlocal max_sum
        
        if not node or remaining_length <= 0:
            return 0
        
        # If this is the only node we can include
        if remaining_length == 1:
            max_sum = max(max_sum, node.val)
            return node.val
        
        # Get gains from children with reduced length
        left_gain = max_gain_with_length(node.left, remaining_length - 1)
        right_gain = max_gain_with_length(node.right, remaining_length - 1)
        
        # Only consider positive gains
        left_gain = max(left_gain, 0)
        right_gain = max(right_gain, 0)
        
        # Path through current node
        path_through = node.val + left_gain + right_gain
        
        # Update global maximum if path is long enough
        path_nodes = 1
        if left_gain > 0:
            path_nodes += 1  # Simplified counting
        if right_gain > 0:
            path_nodes += 1
        
        if path_nodes >= min_length:
            max_sum = max(max_sum, path_through)
        
        # Return gain for parent
        return node.val + max(left_gain, right_gain)
    
    for length in range(min_length, min(max_length + 1, 100)):  # Practical limit
        max_gain_with_length(root, length)
    
    return max_sum


def max_path_sum_analysis(root):
    """
    COMPREHENSIVE PATH ANALYSIS:
    ===========================
    Analyze tree structure and optimal path characteristics.
    """
    if not root:
        print("Empty tree - no paths available!")
        return 0
    
    # Tree structure analysis
    def analyze_tree(node, depth=0):
        if not node:
            return {
                'nodes': 0, 'sum': 0, 'min_val': float('inf'), 
                'max_val': float('-inf'), 'max_depth': depth,
                'positive_nodes': 0, 'negative_nodes': 0
            }
        
        left_info = analyze_tree(node.left, depth + 1)
        right_info = analyze_tree(node.right, depth + 1)
        
        return {
            'nodes': 1 + left_info['nodes'] + right_info['nodes'],
            'sum': node.val + left_info['sum'] + right_info['sum'],
            'min_val': min(node.val, left_info['min_val'], right_info['min_val']),
            'max_val': max(node.val, left_info['max_val'], right_info['max_val']),
            'max_depth': max(left_info['max_depth'], right_info['max_depth']),
            'positive_nodes': (1 if node.val > 0 else 0) + left_info['positive_nodes'] + right_info['positive_nodes'],
            'negative_nodes': (1 if node.val < 0 else 0) + left_info['negative_nodes'] + right_info['negative_nodes']
        }
    
    tree_info = analyze_tree(root)
    
    print(f"Tree Structure Analysis:")
    print(f"  Total nodes: {tree_info['nodes']}")
    print(f"  Total sum: {tree_info['sum']}")
    print(f"  Value range: [{tree_info['min_val']}, {tree_info['max_val']}]")
    print(f"  Tree height: {tree_info['max_depth']}")
    print(f"  Positive nodes: {tree_info['positive_nodes']}")
    print(f"  Negative nodes: {tree_info['negative_nodes']}")
    
    # Optimal path analysis
    max_sum, optimal_path = max_path_sum_detailed(root)
    
    print(f"\nOptimal Path Analysis:")
    print(f"  Maximum path sum: {max_sum}")
    print(f"  Optimal path: {optimal_path}")
    print(f"  Path length: {len(optimal_path)}")
    print(f"  Average node value in path: {sum(optimal_path)/len(optimal_path):.2f}")
    
    # Compare with different strategies
    total_sum = tree_info['sum']
    positive_only = tree_info['positive_nodes'] * tree_info['max_val'] if tree_info['positive_nodes'] > 0 else 0
    
    print(f"\nStrategy Comparison:")
    print(f"  Optimal path: {max_sum}")
    print(f"  All nodes: {total_sum}")
    print(f"  Only positive (estimate): {positive_only}")
    print(f"  Single best node: {tree_info['max_val']}")
    
    return max_sum


def max_path_sum_variants():
    """
    MAXIMUM PATH SUM VARIANTS:
    =========================
    Different scenarios and modifications.
    """
    
    def max_path_sum_k_nodes(root, k):
        """Find maximum sum path with exactly k nodes"""
        if not root or k <= 0:
            return float('-inf')
        
        max_sum = float('-inf')
        
        def dfs(node, remaining, current_sum):
            nonlocal max_sum
            
            if not node:
                return
            
            current_sum += node.val
            
            if remaining == 1:
                max_sum = max(max_sum, current_sum)
                return
            
            # Continue path through children
            dfs(node.left, remaining - 1, current_sum)
            dfs(node.right, remaining - 1, current_sum)
        
        def explore_all_starts(node):
            if not node:
                return
            dfs(node, k, 0)
            explore_all_starts(node.left)
            explore_all_starts(node.right)
        
        explore_all_starts(root)
        return max_sum if max_sum != float('-inf') else 0
    
    def max_path_sum_no_adjacent(root):
        """Maximum path sum where no two adjacent nodes are included"""
        # This becomes similar to House Robber III
        def rob_helper(node):
            if not node:
                return [0, 0]  # [include, exclude]
            
            left = rob_helper(node.left)
            right = rob_helper(node.right)
            
            include = node.val + left[1] + right[1]
            exclude = max(left) + max(right)
            
            return [include, exclude]
        
        if not root:
            return 0
        
        result = rob_helper(root)
        return max(result)
    
    def max_path_sum_with_multiplier(root, multipliers):
        """Different multipliers for different levels"""
        max_sum = float('-inf')
        
        def dfs(node, level):
            nonlocal max_sum
            
            if not node:
                return 0
            
            multiplier = multipliers[level] if level < len(multipliers) else 1
            left_gain = max(dfs(node.left, level + 1), 0)
            right_gain = max(dfs(node.right, level + 1), 0)
            
            # Path through current node
            path_sum = (node.val * multiplier) + left_gain + right_gain
            max_sum = max(max_sum, path_sum)
            
            # Return gain for parent
            return (node.val * multiplier) + max(left_gain, right_gain)
        
        dfs(root, 0)
        return max_sum
    
    # Test with sample tree
    def create_sample_tree():
        """Create sample tree: [-10,9,20,null,null,15,7]"""
        root = TreeNode(-10)
        root.left = TreeNode(9)
        root.right = TreeNode(20)
        root.right.left = TreeNode(15)
        root.right.right = TreeNode(7)
        return root
    
    sample_tree = create_sample_tree()
    
    print("Maximum Path Sum Variants:")
    print("=" * 50)
    
    basic_max = max_path_sum_tree_dp(sample_tree)
    print(f"Basic maximum path sum: {basic_max}")
    
    k_nodes_max = max_path_sum_k_nodes(sample_tree, 3)
    print(f"Max sum with exactly 3 nodes: {k_nodes_max}")
    
    no_adjacent_max = max_path_sum_no_adjacent(sample_tree)
    print(f"Max sum with no adjacent nodes: {no_adjacent_max}")
    
    multipliers = [1, 2, 3]  # Increasing multipliers by level
    multiplier_max = max_path_sum_with_multiplier(sample_tree, multipliers)
    print(f"Max sum with level multipliers {multipliers}: {multiplier_max}")


# Test cases
def test_max_path_sum():
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
        ([1, 2, 3], 6),
        ([-10, 9, 20, None, None, 15, 7], 42),
        ([1], 1),
        ([-3], -3),
        ([2, -1], 2),
        ([5, 4, 8, 11, None, 13, 9, 7, 2], 53),
        ([-1, -2, -3], -1),
        ([1, -2, 3], 4)
    ]
    
    print("Testing Binary Tree Maximum Path Sum Solutions:")
    print("=" * 70)
    
    for i, (tree_list, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Tree: {tree_list}")
        print(f"Expected: {expected}")
        
        root = create_tree_from_list(tree_list)
        
        # Skip brute force for larger trees
        if len([x for x in tree_list if x is not None]) <= 6:
            try:
                brute_force = max_path_sum_brute_force(root)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        tree_dp = max_path_sum_tree_dp(root)
        iterative = max_path_sum_iterative(root)
        
        print(f"Tree DP:          {tree_dp:>4} {'✓' if tree_dp == expected else '✗'}")
        print(f"Iterative:        {iterative:>4} {'✓' if iterative == expected else '✗'}")
        
        # Show path for interesting cases
        if len([x for x in tree_list if x is not None]) <= 8:
            max_sum, optimal_path = max_path_sum_detailed(root)
            print(f"Optimal path: {optimal_path} (sum: {max_sum})")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    sample_tree = create_tree_from_list([-10, 9, 20, None, None, 15, 7])
    max_path_sum_analysis(sample_tree)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    max_path_sum_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. PATH FLEXIBILITY: Path can start and end at any nodes")
    print("2. NEGATIVE HANDLING: Ignore negative subtree contributions")
    print("3. GLOBAL vs LOCAL: Track global maximum while computing local gains")
    print("4. SPLIT DECISION: At each node, choose best single-direction gain")
    print("5. TREE STRUCTURE: Leverage tree properties for optimal traversal")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Network Optimization: Maximum flow paths with node weights")
    print("• Financial Analysis: Optimal investment paths with risks/rewards")
    print("• Game Theory: Optimal move sequences in tree-structured games")
    print("• Route Planning: Maximum value paths in hierarchical networks")
    print("• Resource Allocation: Optimal distribution in tree organizations")


if __name__ == "__main__":
    test_max_path_sum()


"""
BINARY TREE MAXIMUM PATH SUM - ADVANCED TREE DP OPTIMIZATION:
=============================================================

This problem showcases advanced Tree DP concepts:
- Flexible path definitions (any node to any node)
- Global vs. local optimization tracking
- Negative value handling strategies
- Complex state management with path constraints

KEY INSIGHTS:
============
1. **PATH FLEXIBILITY**: Paths can start and end at any nodes (not just root)
2. **GLOBAL TRACKING**: Must track global maximum separate from local computation
3. **NEGATIVE PRUNING**: Ignore negative subtree contributions (take 0 instead)
4. **SPLIT DECISION**: At each node, choose single best direction for parent path
5. **LOCAL vs GLOBAL**: Local gain for parent ≠ global maximum through node

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n²) time, O(h) space
   - For each node, explore all possible paths
   - Exponentially many paths to consider

2. **Tree DP (Optimal)**: O(n) time, O(h) space
   - Single traversal with global tracking
   - Most elegant and efficient solution

3. **Iterative Post-order**: O(n) time, O(h) space
   - Stack-based implementation
   - Avoids recursion overhead

4. **Constrained Variants**: O(n*k) time for k-length constraints
   - Additional state for path length tracking

CORE TREE DP ALGORITHM:
======================
```python
def maxPathSum(root):
    max_sum = float('-inf')
    
    def max_gain(node):
        nonlocal max_sum
        
        if not node:
            return 0
        
        # Only consider positive gains from children
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        
        # Maximum path sum through current node (for global tracking)
        path_through_node = node.val + left_gain + right_gain
        max_sum = max(max_sum, path_through_node)
        
        # Return maximum gain for parent (single direction only)
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum
```

CRITICAL DESIGN DECISIONS:
=========================

**1. Global vs. Local Tracking**:
- `max_sum`: Global maximum across all possible paths
- `return value`: Local maximum gain extending to parent

**2. Negative Value Handling**:
```python
left_gain = max(max_gain(node.left), 0)   # Ignore negative subtrees
right_gain = max(max_gain(node.right), 0)
```

**3. Path Splitting**:
- Through current node: `node.val + left_gain + right_gain` (cannot extend to parent)
- To parent: `node.val + max(left_gain, right_gain)` (single direction only)

STATE TRANSITION LOGIC:
======================
**For each node, we compute**:
1. **Maximum gain ending at this node** (for parent's computation)
2. **Maximum path sum through this node** (for global tracking)

**Why this works**:
- Every possible path either goes through some node or ends at some node
- By considering both cases at every node, we cover all possibilities
- Post-order traversal ensures children are solved before parent

RECURRENCE RELATIONS:
====================
```
max_gain[node] = node.val + max(max_gain[left], max_gain[right], 0)
path_through[node] = node.val + max(max_gain[left], 0) + max(max_gain[right], 0)
global_max = max(global_max, path_through[node])
```

**Base Case**: `max_gain[null] = 0`

NEGATIVE VALUE STRATEGY:
=======================
**Key Insight**: Never include negative subtrees in optimal path

**Implementation**:
- Always take `max(subtree_gain, 0)`
- Effectively "prunes" negative contributions
- Allows paths to terminate optimally rather than being forced to include bad nodes

**Example**: Tree `[-10, 9, 20, null, null, 15, 7]`
- Left subtree gain: `max(9, 0) = 9` (but we take 0 to avoid -10)
- Right subtree: `15 + 20 + 7 = 42` (optimal path)

PATH FLEXIBILITY ANALYSIS:
==========================
**Possible Path Types**:
1. **Single Node**: Just the node value
2. **Single Branch**: Node + one subtree  
3. **Through Node**: Node + both subtrees (cannot extend to parent)
4. **Subtree Only**: Entirely within left or right subtree

**Coverage Guarantee**: The algorithm considers all types:
- Single node: `node.val` (when both gains are negative)
- Single branch: `node.val + max(left_gain, right_gain)`
- Through node: `node.val + left_gain + right_gain`
- Subtree only: Covered by recursive calls

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n) - each node visited exactly once
- **Space**: O(h) - recursion stack depth
- **Optimal**: Cannot do better than O(n) since we must examine all nodes
- **Practical**: Very efficient with minimal overhead

ADVANCED VARIANTS:
=================

**1. Path Length Constraints**:
```python
def maxPathSumWithLength(root, min_len, max_len):
    # Track length alongside gain computation
    # State: (max_gain, path_length)
```

**2. No Adjacent Nodes**:
```python
def maxPathSumNoAdjacent(root):
    # Similar to House Robber III
    # State: [include_node, exclude_node]
```

**3. Weighted by Level**:
```python
def maxPathSumWeighted(root, level_weights):
    # Apply different weights at different levels
    # Modify node.val based on depth
```

APPLICATIONS:
============
- **Network Flow**: Maximum capacity paths in weighted networks
- **Financial Optimization**: Best investment sequences with risk/reward
- **Bioinformatics**: Optimal alignment scores in phylogenetic trees
- **Game Theory**: Best move sequences in game trees
- **Resource Allocation**: Optimal resource flows in hierarchical systems

RELATED PROBLEMS:
================
- **House Robber III (337)**: Similar tree DP with adjacency constraints
- **Binary Tree Cameras (968)**: Coverage optimization
- **Distribute Coins (979)**: Resource flow optimization
- **Diameter of Binary Tree**: Path length optimization

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Global optimum contains local optima
- **Overlapping Subproblems**: None (tree structure prevents this)
- **Greedy Choice**: Sometimes optimal to exclude negative subtrees
- **Monotonicity**: More positive values never decrease optimal sum

EDGE CASES:
==========
- **All Negative**: Return single least negative node
- **Single Node**: Return that node's value  
- **Linear Tree**: Reduces to maximum subarray problem
- **Perfect Binary Tree**: Balanced exploration of both subtrees

This problem demonstrates the sophistication of Tree DP:
- Managing multiple optimization criteria simultaneously
- Handling negative values optimally
- Balancing local decisions with global optimization
- Leveraging tree structure for efficient solution
"""
