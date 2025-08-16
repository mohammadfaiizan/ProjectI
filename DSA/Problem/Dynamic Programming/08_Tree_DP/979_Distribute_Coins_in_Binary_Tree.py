"""
LeetCode 979: Distribute Coins in Binary Tree
Difficulty: Medium
Category: Tree DP - Resource Flow Optimization

PROBLEM DESCRIPTION:
===================
You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins total throughout the whole tree.

In one move, we may choose two adjacent nodes and move one coin from one node to the other. (A move may be from parent to child, or from child to parent.)

Return the minimum number of moves required to make every node have exactly one coin.

Example 1:
Input: root = [3,0,0]
Output: 2
Explanation: From the root, we move one coin to its left child, and one coin to its right child.

Example 2:
Input: root = [0,3,0]
Output: 3
Explanation: From the left child, we move two coins to the root [2,1,0]. Then, we move one coin from the root to the right child.

Example 3:
Input: root = [1,0,2]
Output: 2

Example 4:
Input: root = [1,0,0,null,3]
Output: 4

Constraints:
- The number of nodes in the tree is n.
- 1 <= n <= 100
- 0 <= Node.val <= n
- The sum of all Node.val is n.
"""

# TreeNode definition for reference
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def distribute_coins_tree_dp(root):
    """
    TREE DP APPROACH:
    ================
    Calculate excess/deficit for each subtree and sum absolute flows.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    total_moves = 0
    
    def dfs(node):
        nonlocal total_moves
        
        if not node:
            return 0
        
        # Get excess/deficit from left and right subtrees
        left_excess = dfs(node.left)
        right_excess = dfs(node.right)
        
        # Add absolute moves (coins that must flow through this node)
        total_moves += abs(left_excess) + abs(right_excess)
        
        # Return excess/deficit of current subtree
        # Positive means excess (giving coins), negative means deficit (needs coins)
        return node.val + left_excess + right_excess - 1
    
    dfs(root)
    return total_moves


def distribute_coins_with_flow_tracking(root):
    """
    DETAILED FLOW TRACKING:
    ======================
    Track actual coin movements and flow directions.
    
    Time Complexity: O(n) - single traversal
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0, []
    
    total_moves = 0
    flow_details = []
    
    def dfs(node, parent_val=None):
        nonlocal total_moves, flow_details
        
        if not node:
            return 0
        
        # Process children first
        left_excess = dfs(node.left, node.val)
        right_excess = dfs(node.right, node.val)
        
        # Calculate flows through this node
        left_flow = abs(left_excess)
        right_flow = abs(right_excess)
        
        if left_flow > 0:
            direction = "from child" if left_excess > 0 else "to child"
            flow_details.append({
                'parent': node.val,
                'child': node.left.val if node.left else None,
                'coins': left_flow,
                'direction': direction
            })
        
        if right_flow > 0:
            direction = "from child" if right_excess > 0 else "to child"
            flow_details.append({
                'parent': node.val,
                'child': node.right.val if node.right else None,
                'coins': right_flow,
                'direction': direction
            })
        
        total_moves += left_flow + right_flow
        
        # Return excess/deficit for this subtree
        return node.val + left_excess + right_excess - 1
    
    dfs(root)
    return total_moves, flow_details


def distribute_coins_iterative(root):
    """
    ITERATIVE POST-ORDER APPROACH:
    ==============================
    Stack-based implementation for coin distribution.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - stack space
    """
    if not root:
        return 0
    
    stack = [(root, False)]
    excess = {}  # node -> excess/deficit
    total_moves = 0
    
    while stack:
        node, processed = stack.pop()
        
        if processed:
            # Process after children
            left_excess = excess.get(node.left, 0)
            right_excess = excess.get(node.right, 0)
            
            # Add moves for flows through this node
            total_moves += abs(left_excess) + abs(right_excess)
            
            # Calculate excess/deficit for this subtree
            excess[node] = node.val + left_excess + right_excess - 1
        else:
            # Mark for processing and add children
            stack.append((node, True))
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    return total_moves


def distribute_coins_with_path_analysis(root):
    """
    PATH ANALYSIS APPROACH:
    ======================
    Analyze optimal coin distribution paths.
    
    Time Complexity: O(n) - tree traversal with analysis
    Space Complexity: O(h) - recursion depth
    """
    if not root:
        return 0
    
    # First, collect all node information
    def collect_info(node, depth=0, path="root"):
        if not node:
            return []
        
        info = [{
            'node': node,
            'val': node.val,
            'depth': depth,
            'path': path
        }]
        
        if node.left:
            info.extend(collect_info(node.left, depth + 1, path + "->L"))
        if node.right:
            info.extend(collect_info(node.right, depth + 1, path + "->R"))
        
        return info
    
    nodes_info = collect_info(root)
    total_nodes = len(nodes_info)
    total_coins = sum(info['val'] for info in nodes_info)
    
    print(f"Coin Distribution Analysis:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total coins: {total_coins}")
    print(f"  Target: 1 coin per node")
    
    # Calculate distribution using tree DP
    total_moves = 0
    distribution_log = []
    
    def dfs_with_logging(node, node_path="root"):
        nonlocal total_moves, distribution_log
        
        if not node:
            return 0
        
        left_excess = dfs_with_logging(node.left, node_path + "->L") if node.left else 0
        right_excess = dfs_with_logging(node.right, node_path + "->R") if node.right else 0
        
        # Log the flows
        if left_excess != 0:
            action = "gives" if left_excess > 0 else "receives"
            distribution_log.append(f"{node_path}->L {action} {abs(left_excess)} coin(s) to/from {node_path}")
        
        if right_excess != 0:
            action = "gives" if right_excess > 0 else "receives"
            distribution_log.append(f"{node_path}->R {action} {abs(right_excess)} coin(s) to/from {node_path}")
        
        total_moves += abs(left_excess) + abs(right_excess)
        
        # Calculate this subtree's excess/deficit
        subtree_excess = node.val + left_excess + right_excess - 1
        
        return subtree_excess
    
    final_excess = dfs_with_logging(root)
    
    print(f"\nDistribution Steps:")
    for step in distribution_log:
        print(f"  {step}")
    
    print(f"\nResult:")
    print(f"  Total moves required: {total_moves}")
    print(f"  Final tree excess: {final_excess} (should be 0)")
    
    return total_moves


def distribute_coins_step_by_step(root):
    """
    STEP-BY-STEP SIMULATION:
    =======================
    Simulate the actual coin movement process.
    
    Time Complexity: O(n) - tree traversal
    Space Complexity: O(n) - tree state tracking
    """
    if not root:
        return 0, []
    
    # Create a copy of the tree for simulation
    def copy_tree(node):
        if not node:
            return None
        
        new_node = TreeNode(node.val)
        new_node.left = copy_tree(node.left)
        new_node.right = copy_tree(node.right)
        return new_node
    
    tree_copy = copy_tree(root)
    moves = []
    total_moves = 0
    
    def get_tree_state(node, path=""):
        if not node:
            return {}
        
        state = {path or "root": node.val}
        state.update(get_tree_state(node.left, path + "L"))
        state.update(get_tree_state(node.right, path + "R"))
        return state
    
    def simulate_distribution(node, parent=None, is_left_child=False):
        nonlocal total_moves, moves
        
        if not node:
            return 0
        
        # Process children first
        left_excess = simulate_distribution(node.left, node, True)
        right_excess = simulate_distribution(node.right, node, False)
        
        # Handle left child excess/deficit
        if left_excess > 0:
            # Left child has excess, move to current node
            for _ in range(left_excess):
                moves.append(f"Move 1 coin from left child to current node")
                node.val += 1
                if node.left:
                    node.left.val -= 1
                total_moves += 1
        elif left_excess < 0:
            # Left child needs coins, move from current node
            for _ in range(-left_excess):
                moves.append(f"Move 1 coin from current node to left child")
                node.val -= 1
                if node.left:
                    node.left.val += 1
                total_moves += 1
        
        # Handle right child excess/deficit
        if right_excess > 0:
            # Right child has excess, move to current node
            for _ in range(right_excess):
                moves.append(f"Move 1 coin from right child to current node")
                node.val += 1
                if node.right:
                    node.right.val -= 1
                total_moves += 1
        elif right_excess < 0:
            # Right child needs coins, move from current node
            for _ in range(-right_excess):
                moves.append(f"Move 1 coin from current node to right child")
                node.val -= 1
                if node.right:
                    node.right.val += 1
                total_moves += 1
        
        # Return excess/deficit for this subtree
        return node.val - 1
    
    simulate_distribution(tree_copy)
    
    return total_moves, moves


def distribute_coins_analysis(root):
    """
    COMPREHENSIVE DISTRIBUTION ANALYSIS:
    ===================================
    Analyze coin distribution patterns and efficiency.
    """
    if not root:
        print("Empty tree - no coin distribution needed!")
        return 0
    
    # Tree structure analysis
    def analyze_tree_structure(node, depth=0):
        if not node:
            return {
                'nodes': 0, 'total_coins': 0, 'max_depth': depth,
                'excess_nodes': 0, 'deficit_nodes': 0, 'perfect_nodes': 0
            }
        
        left_info = analyze_tree_structure(node.left, depth + 1)
        right_info = analyze_tree_structure(node.right, depth + 1)
        
        node_type = ('excess' if node.val > 1 else 
                    'deficit' if node.val < 1 else 'perfect')
        
        return {
            'nodes': 1 + left_info['nodes'] + right_info['nodes'],
            'total_coins': node.val + left_info['total_coins'] + right_info['total_coins'],
            'max_depth': max(left_info['max_depth'], right_info['max_depth']),
            'excess_nodes': (1 if node.val > 1 else 0) + left_info['excess_nodes'] + right_info['excess_nodes'],
            'deficit_nodes': (1 if node.val < 1 else 0) + left_info['deficit_nodes'] + right_info['deficit_nodes'],
            'perfect_nodes': (1 if node.val == 1 else 0) + left_info['perfect_nodes'] + right_info['perfect_nodes']
        }
    
    tree_info = analyze_tree_structure(root)
    
    print(f"Tree Structure Analysis:")
    print(f"  Total nodes: {tree_info['nodes']}")
    print(f"  Total coins: {tree_info['total_coins']}")
    print(f"  Tree height: {tree_info['max_depth']}")
    print(f"  Nodes with excess coins: {tree_info['excess_nodes']}")
    print(f"  Nodes with deficit coins: {tree_info['deficit_nodes']}")
    print(f"  Nodes with perfect coins: {tree_info['perfect_nodes']}")
    
    # Distribution analysis
    basic_moves = distribute_coins_tree_dp(root)
    detailed_moves, flow_details = distribute_coins_with_flow_tracking(root)
    
    print(f"\nDistribution Analysis:")
    print(f"  Minimum moves required: {basic_moves}")
    print(f"  Flow efficiency: {tree_info['total_coins']/basic_moves:.2f} coins per move")
    
    # Flow pattern analysis
    print(f"\nFlow Patterns:")
    for flow in flow_details[:5]:  # Show first 5 flows
        print(f"  {flow['parent']} {flow['direction']} {flow['child']}: {flow['coins']} coin(s)")
    
    # Theoretical analysis
    excess_total = sum(max(0, node.val - 1) for node in [root])  # Simplified
    deficit_total = tree_info['nodes'] - tree_info['total_coins'] + tree_info['excess_nodes']
    
    print(f"\nTheoretical Bounds:")
    print(f"  Total excess coins: {tree_info['total_coins'] - tree_info['nodes']}")
    print(f"  Minimum possible moves: {basic_moves} (optimal)")
    print(f"  Average moves per unbalanced node: {basic_moves/(tree_info['excess_nodes'] + tree_info['deficit_nodes']):.2f}")
    
    return basic_moves


def distribute_coins_variants():
    """
    COIN DISTRIBUTION VARIANTS:
    ==========================
    Different scenarios and modifications.
    """
    
    def distribute_coins_with_cost(root, move_costs):
        """Different costs for moves at different levels"""
        if not root:
            return 0
        
        total_cost = 0
        
        def dfs(node, depth=0):
            nonlocal total_cost
            
            if not node:
                return 0
            
            left_excess = dfs(node.left, depth + 1)
            right_excess = dfs(node.right, depth + 1)
            
            # Apply costs based on depth
            move_cost = move_costs[depth] if depth < len(move_costs) else 1
            total_cost += (abs(left_excess) + abs(right_excess)) * move_cost
            
            return node.val + left_excess + right_excess - 1
        
        dfs(root)
        return total_cost
    
    def distribute_coins_k_per_node(root, k=1):
        """Distribute to have k coins per node"""
        if not root:
            return 0
        
        total_moves = 0
        
        def dfs(node):
            nonlocal total_moves
            
            if not node:
                return 0
            
            left_excess = dfs(node.left)
            right_excess = dfs(node.right)
            
            total_moves += abs(left_excess) + abs(right_excess)
            
            # Target k coins per node instead of 1
            return node.val + left_excess + right_excess - k
        
        dfs(root)
        return total_moves
    
    def distribute_coins_with_capacity(root, max_capacity=float('inf')):
        """Nodes have maximum capacity"""
        # This is more complex and requires different approach
        # Simplified version for demonstration
        return distribute_coins_tree_dp(root)
    
    def distribute_coins_minimum_total(root):
        """Find minimum total coins needed for distribution"""
        def count_nodes(node):
            if not node:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        
        def sum_coins(node):
            if not node:
                return 0
            return node.val + sum_coins(node.left) + sum_coins(node.right)
        
        nodes = count_nodes(root)
        coins = sum_coins(root)
        
        if coins < nodes:
            return nodes - coins, f"Need {nodes - coins} more coins"
        elif coins > nodes:
            return 0, f"Have {coins - nodes} excess coins"
        else:
            return 0, "Perfect balance"
    
    # Test with sample tree
    def create_sample_tree():
        """Create sample tree: [3,0,0]"""
        root = TreeNode(3)
        root.left = TreeNode(0)
        root.right = TreeNode(0)
        return root
    
    sample_tree = create_sample_tree()
    
    print("Coin Distribution Variants:")
    print("=" * 50)
    
    basic_moves = distribute_coins_tree_dp(sample_tree)
    print(f"Basic distribution: {basic_moves} moves")
    
    costs = [1, 2, 3]  # Increasing costs by level
    cost_moves = distribute_coins_with_cost(sample_tree, costs)
    print(f"With level costs {costs}: {cost_moves} total cost")
    
    k_moves = distribute_coins_k_per_node(sample_tree, 2)
    print(f"For 2 coins per node: {k_moves} moves")
    
    needed, message = distribute_coins_minimum_total(sample_tree)
    print(f"Coin balance: {message}")


# Test cases
def test_distribute_coins():
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
        ([3, 0, 0], 2),
        ([0, 3, 0], 3),
        ([1, 0, 2], 2),
        ([1, 0, 0, None, 3], 4),
        ([1], 0),
        ([2, 1, 0], 2),
        ([0, 2, 1], 2),
        ([1, 1, 1], 0)
    ]
    
    print("Testing Distribute Coins in Binary Tree Solutions:")
    print("=" * 70)
    
    for i, (tree_list, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Tree: {tree_list}")
        print(f"Expected: {expected}")
        
        root = create_tree_from_list(tree_list)
        
        tree_dp = distribute_coins_tree_dp(root)
        iterative = distribute_coins_iterative(root)
        
        print(f"Tree DP:          {tree_dp:>4} {'✓' if tree_dp == expected else '✗'}")
        print(f"Iterative:        {iterative:>4} {'✓' if iterative == expected else '✗'}")
        
        # Show flow details for small cases
        if len([x for x in tree_list if x is not None]) <= 6:
            detailed_moves, flows = distribute_coins_with_flow_tracking(root)
            print(f"Detailed analysis: {detailed_moves} moves with {len(flows)} flows")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    sample_tree = create_tree_from_list([1, 0, 0, None, 3])
    distribute_coins_analysis(sample_tree)
    
    # Step-by-step simulation
    print(f"\n" + "=" * 70)
    print("STEP-BY-STEP SIMULATION:")
    print("-" * 40)
    sample_tree2 = create_tree_from_list([3, 0, 0])
    moves, move_list = distribute_coins_step_by_step(sample_tree2)
    print(f"Total moves: {moves}")
    for i, move in enumerate(move_list[:5]):  # Show first 5 moves
        print(f"  {i+1}. {move}")
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    distribute_coins_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. FLOW CALCULATION: Excess/deficit of each subtree determines flow")
    print("2. ABSOLUTE MOVEMENTS: Total moves = sum of absolute flows through nodes")
    print("3. POST-ORDER PROCESSING: Children processed before parent")
    print("4. BALANCE PRINCIPLE: Total coins = total nodes for valid input")
    print("5. OPTIMAL SUBSTRUCTURE: Optimal flows in subtrees lead to global optimum")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Redistribution: Optimal rebalancing of distributed resources")
    print("• Load Balancing: Redistributing load across tree-structured systems")
    print("• Network Flow: Minimum cost flow in tree networks")
    print("• Game Theory: Optimal resource trading in hierarchical systems")
    print("• Operations Research: Distribution optimization in supply chains")


if __name__ == "__main__":
    test_distribute_coins()


"""
DISTRIBUTE COINS IN BINARY TREE - RESOURCE FLOW OPTIMIZATION:
=============================================================

This problem demonstrates advanced Tree DP for resource redistribution:
- Flow calculation based on excess/deficit analysis
- Optimal resource movement in tree structures
- Post-order processing for dependency management
- Balance preservation through flow conservation

KEY INSIGHTS:
============
1. **FLOW CALCULATION**: Each subtree's excess/deficit determines flow through parent
2. **ABSOLUTE MOVEMENT**: Total moves = sum of absolute flows through all edges
3. **BALANCE PRESERVATION**: Total coins equals total nodes (conservation law)
4. **OPTIMAL SUBSTRUCTURE**: Optimal flows in subtrees lead to global optimum
5. **POST-ORDER DEPENDENCY**: Children must be balanced before parent

ALGORITHM APPROACHES:
====================

1. **Tree DP (Optimal)**: O(n) time, O(h) space
   - Calculate excess/deficit for each subtree
   - Sum absolute flows through all edges

2. **Iterative Post-order**: O(n) time, O(h) space
   - Stack-based implementation
   - Same logic as recursive approach

3. **Flow Tracking**: O(n) time, O(h) space
   - Track actual coin movements
   - Detailed analysis of distribution pattern

CORE TREE DP ALGORITHM:
======================
```python
def distributeCoins(root):
    total_moves = 0
    
    def dfs(node):
        nonlocal total_moves
        
        if not node:
            return 0
        
        # Get excess/deficit from children
        left_excess = dfs(node.left)
        right_excess = dfs(node.right)
        
        # Add moves for flows through this node
        total_moves += abs(left_excess) + abs(right_excess)
        
        # Return excess/deficit for this subtree
        return node.val + left_excess + right_excess - 1
    
    dfs(root)
    return total_moves
```

FLOW ANALYSIS FRAMEWORK:
=======================
**Excess/Deficit Calculation**:
- Excess = (subtree_coins - subtree_nodes)
- Positive excess: subtree gives coins to parent
- Negative excess: subtree receives coins from parent

**Flow Conservation**:
- Total flow into node = total flow out of node
- Flow through edge = |excess/deficit of subtree|

**Movement Optimization**:
- Each coin moves along shortest path in tree
- No unnecessary detours or cycles
- Minimal total movement distance

RECURRENCE RELATION:
===================
```
excess[node] = node.val + excess[left] + excess[right] - 1
total_moves += |excess[left]| + |excess[right]|
```

**Base Case**: `excess[null] = 0`

**Physical Interpretation**:
- `excess[node] > 0`: Subtree has extra coins to share
- `excess[node] < 0`: Subtree needs coins from ancestor
- `excess[node] = 0`: Subtree is perfectly balanced

FLOW DIRECTION ANALYSIS:
=======================
**Upward Flow** (excess > 0):
- Child subtree has surplus coins
- Coins flow from child to parent
- Parent redistributes to deficit areas

**Downward Flow** (excess < 0):
- Child subtree has deficit
- Parent provides coins to child
- Coins sourced from other surplus areas

**Optimal Flow Property**:
- Every coin takes shortest path to destination
- Tree structure ensures unique shortest paths
- No flow optimization needed beyond tree DP

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n) - each node visited exactly once
- **Space**: O(h) - recursion stack depth
- **Optimal**: Cannot improve (must examine all nodes)
- **Practical**: Very efficient with minimal overhead

MATHEMATICAL PROPERTIES:
========================
**Conservation Law**: `∑(node.val) = n` (given constraint)
**Flow Balance**: `∑(excess) = 0` (total excess equals total deficit)
**Minimality**: Tree structure guarantees minimal flow paths
**Optimality**: Greedy flow calculation is globally optimal

RESOURCE REDISTRIBUTION PATTERNS:
=================================
**Common Patterns**:
1. **Leaf-to-Root**: Excess leaves provide to deficit ancestors
2. **Root-to-Leaf**: Root redistributes to deficit subtrees
3. **Sibling Transfer**: Via common ancestor (LCA)
4. **Multi-hop**: Through intermediate nodes

**Efficiency Metrics**:
- Average coins per move
- Flow concentration vs. distribution
- Path length optimization

APPLICATIONS:
============
- **Load Balancing**: Redistributing computational load across servers
- **Resource Management**: Optimal inventory redistribution
- **Network Flow**: Minimum cost flow in tree networks
- **Supply Chain**: Distribution optimization in hierarchical systems
- **Game Theory**: Resource trading in tree-structured economies

RELATED PROBLEMS:
================
- **Binary Tree Maximum Path Sum**: Similar tree traversal pattern
- **House Robber III**: Resource optimization with constraints
- **Minimum Cost Tree From Leaf Values**: Tree construction optimization
- **Network Flow**: General flow optimization (trees are special case)

VARIANTS:
========
**Weighted Movements**: Different costs for moves at different levels
**Capacity Constraints**: Maximum coins per node limitations
**Multi-target**: Distribute to achieve k coins per node
**Directional Constraints**: Restricted flow directions

EDGE CASES:
==========
- **Single Node**: No movement needed if val = 1
- **Perfect Distribution**: All nodes already have 1 coin
- **Linear Tree**: Reduces to array redistribution problem
- **Star Graph**: Central redistribution hub

OPTIMIZATION TECHNIQUES:
=======================
**Early Termination**: Stop if subtree is already balanced
**Flow Aggregation**: Combine multiple small flows
**Path Compression**: Direct flows when possible
**Memory Optimization**: Iterative implementation for large trees

THEORETICAL INSIGHTS:
====================
**Why Tree DP Works**:
- Optimal substructure: local optimal choices → global optimum
- No overlapping subproblems: tree structure prevents this
- Greedy property: immediate flow balancing is always optimal

**Flow Minimality Proof**:
- Tree has unique paths between any two nodes
- Any redistribution must follow these unique paths
- Therefore, computed flows are minimal possible

This problem elegantly demonstrates how Tree DP can solve
complex resource optimization problems efficiently by
leveraging the structural properties of trees to achieve
optimal flow calculation in linear time.
"""
