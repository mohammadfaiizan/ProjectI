"""
LeetCode 337: House Robber III
Difficulty: Medium
Category: Tree DP - Binary Tree Optimization

PROBLEM DESCRIPTION:
===================
The thief has found himself a new place for his thievery: a binary tree representing houses.
All houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

Example 1:
Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

Example 2:
Input: root = [3,4,5,1,3,null,1]
Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.

Constraints:
- The number of nodes in the tree is in the range [0, 10^4].
- 0 <= Node.val <= 10^4
"""

# TreeNode definition for reference
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def rob_recursive_brute_force(root):
    """
    BRUTE FORCE RECURSIVE APPROACH:
    ==============================
    For each node, try both robbing it and not robbing it.
    
    Time Complexity: O(2^n) - exponential due to overlapping subproblems
    Space Complexity: O(h) - recursion stack height
    """
    if not root:
        return 0
    
    # Option 1: Rob current house
    rob_current = root.val
    if root.left:
        rob_current += rob_recursive_brute_force(root.left.left) + rob_recursive_brute_force(root.left.right)
    if root.right:
        rob_current += rob_recursive_brute_force(root.right.left) + rob_recursive_brute_force(root.right.right)
    
    # Option 2: Don't rob current house
    not_rob_current = rob_recursive_brute_force(root.left) + rob_recursive_brute_force(root.right)
    
    return max(rob_current, not_rob_current)


def rob_memoization(root):
    """
    MEMOIZATION APPROACH:
    ====================
    Cache results for each node to avoid recalculation.
    
    Time Complexity: O(n) - each node computed once
    Space Complexity: O(n) - memoization table + O(h) recursion
    """
    memo = {}
    
    def rob_helper(node):
        if not node:
            return 0
        
        if node in memo:
            return memo[node]
        
        # Option 1: Rob current house
        rob_current = node.val
        if node.left:
            rob_current += rob_helper(node.left.left) + rob_helper(node.left.right)
        if node.right:
            rob_current += rob_helper(node.right.left) + rob_helper(node.right.right)
        
        # Option 2: Don't rob current house
        not_rob_current = rob_helper(node.left) + rob_helper(node.right)
        
        result = max(rob_current, not_rob_current)
        memo[node] = result
        return result
    
    return rob_helper(root)


def rob_tree_dp_optimal(root):
    """
    OPTIMAL TREE DP APPROACH:
    ========================
    Return two values for each node: [rob_this, not_rob_this]
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - recursion stack only
    """
    def rob_helper(node):
        if not node:
            return [0, 0]  # [rob_this, not_rob_this]
        
        left_result = rob_helper(node.left)
        right_result = rob_helper(node.right)
        
        # If we rob this node, we cannot rob children
        rob_this = node.val + left_result[1] + right_result[1]
        
        # If we don't rob this node, we can choose best from children
        not_rob_this = max(left_result) + max(right_result)
        
        return [rob_this, not_rob_this]
    
    result = rob_helper(root)
    return max(result)


def rob_tree_dp_detailed(root):
    """
    DETAILED TREE DP WITH STATE TRACKING:
    ====================================
    Track both states and return detailed analysis.
    
    Time Complexity: O(n) - single traversal
    Space Complexity: O(h) - recursion stack
    """
    def rob_helper(node):
        if not node:
            return {
                'rob': 0,
                'not_rob': 0,
                'nodes_robbed': [],
                'nodes_not_robbed': []
            }
        
        left_result = rob_helper(node.left)
        right_result = rob_helper(node.right)
        
        # If we rob this node
        rob_value = node.val + left_result['not_rob'] + right_result['not_rob']
        rob_nodes = [node.val] + left_result['nodes_not_robbed'] + right_result['nodes_not_robbed']
        
        # If we don't rob this node
        left_best = left_result['rob'] if left_result['rob'] > left_result['not_rob'] else left_result['not_rob']
        right_best = right_result['rob'] if right_result['rob'] > right_result['not_rob'] else right_result['not_rob']
        not_rob_value = left_best + right_best
        
        # Choose optimal nodes for not robbing this node
        left_optimal_nodes = (left_result['nodes_robbed'] if left_result['rob'] > left_result['not_rob'] 
                             else left_result['nodes_not_robbed'])
        right_optimal_nodes = (right_result['nodes_robbed'] if right_result['rob'] > right_result['not_rob'] 
                              else right_result['nodes_not_robbed'])
        not_rob_nodes = left_optimal_nodes + right_optimal_nodes
        
        return {
            'rob': rob_value,
            'not_rob': not_rob_value,
            'nodes_robbed': rob_nodes,
            'nodes_not_robbed': not_rob_nodes
        }
    
    if not root:
        return 0
    
    result = rob_helper(root)
    max_value = max(result['rob'], result['not_rob'])
    optimal_nodes = result['nodes_robbed'] if result['rob'] > result['not_rob'] else result['nodes_not_robbed']
    
    return max_value, optimal_nodes


def rob_iterative_postorder(root):
    """
    ITERATIVE APPROACH USING POST-ORDER TRAVERSAL:
    ==============================================
    Use stack-based traversal with state tracking.
    
    Time Complexity: O(n) - each node visited once
    Space Complexity: O(h) - stack space
    """
    if not root:
        return 0
    
    stack = [(root, False)]  # (node, processed)
    results = {}  # node -> [rob, not_rob]
    
    while stack:
        node, processed = stack.pop()
        
        if processed:
            # Process node after children are processed
            left_rob = results.get(node.left, [0, 0])
            right_rob = results.get(node.right, [0, 0])
            
            rob_this = node.val + left_rob[1] + right_rob[1]
            not_rob_this = max(left_rob) + max(right_rob)
            
            results[node] = [rob_this, not_rob_this]
        else:
            # Mark for processing and add children
            stack.append((node, True))
            if node.right:
                stack.append((node.right, False))
            if node.left:
                stack.append((node.left, False))
    
    return max(results[root])


def rob_tree_analysis(root):
    """
    COMPREHENSIVE TREE ANALYSIS:
    ============================
    Analyze tree structure and optimal robbing strategy.
    """
    if not root:
        print("Empty tree - no houses to rob!")
        return 0
    
    # Tree structure analysis
    def analyze_tree(node, depth=0):
        if not node:
            return {'nodes': 0, 'total_value': 0, 'max_depth': depth}
        
        left_info = analyze_tree(node.left, depth + 1)
        right_info = analyze_tree(node.right, depth + 1)
        
        return {
            'nodes': 1 + left_info['nodes'] + right_info['nodes'],
            'total_value': node.val + left_info['total_value'] + right_info['total_value'],
            'max_depth': max(left_info['max_depth'], right_info['max_depth'])
        }
    
    tree_info = analyze_tree(root)
    
    print(f"Tree Analysis:")
    print(f"  Total nodes: {tree_info['nodes']}")
    print(f"  Total house values: {tree_info['total_value']}")
    print(f"  Tree height: {tree_info['max_depth']}")
    
    # Optimal robbing analysis
    max_rob, optimal_nodes = rob_tree_dp_detailed(root)
    
    print(f"\nOptimal Robbing Strategy:")
    print(f"  Maximum money: {max_rob}")
    print(f"  Houses to rob: {optimal_nodes}")
    print(f"  Efficiency: {max_rob/tree_info['total_value']:.2%}")
    
    # Level-by-level analysis
    def level_analysis(node, level=0, levels=None):
        if levels is None:
            levels = {}
        if not node:
            return levels
        
        if level not in levels:
            levels[level] = []
        levels[level].append(node.val)
        
        level_analysis(node.left, level + 1, levels)
        level_analysis(node.right, level + 1, levels)
        return levels
    
    levels = level_analysis(root)
    print(f"\nLevel-by-level house values:")
    for level, values in levels.items():
        print(f"  Level {level}: {values} (sum: {sum(values)})")
    
    return max_rob


def rob_tree_variants():
    """
    HOUSE ROBBER TREE VARIANTS:
    ===========================
    Different scenarios and modifications.
    """
    
    def rob_with_constraints(root, max_houses):
        """Rob at most max_houses"""
        def rob_helper(node, houses_used):
            if not node or houses_used > max_houses:
                return 0
            
            # Option 1: Rob this house
            rob_this = (node.val + 
                       rob_helper(node.left, houses_used + 1) + 
                       rob_helper(node.right, houses_used + 1))
            
            # Option 2: Don't rob this house
            not_rob_this = (rob_helper(node.left, houses_used) + 
                           rob_helper(node.right, houses_used))
            
            return max(rob_this, not_rob_this)
        
        return rob_helper(root, 0)
    
    def rob_with_costs(root, costs):
        """Rob with different costs for different levels"""
        def rob_helper(node, level=0):
            if not node:
                return [0, 0]
            
            left_result = rob_helper(node.left, level + 1)
            right_result = rob_helper(node.right, level + 1)
            
            cost = costs[level] if level < len(costs) else 1
            rob_this = node.val - cost + left_result[1] + right_result[1]
            not_rob_this = max(left_result) + max(right_result)
            
            return [rob_this, not_rob_this]
        
        result = rob_helper(root)
        return max(result) if result else 0
    
    def rob_tree_with_guards(root, guard_positions):
        """Some houses have guards (cannot be robbed)"""
        def rob_helper(node, position=""):
            if not node:
                return [0, 0]
            
            left_result = rob_helper(node.left, position + "L")
            right_result = rob_helper(node.right, position + "R")
            
            if position in guard_positions:
                # Cannot rob this house
                rob_this = 0
            else:
                rob_this = node.val + left_result[1] + right_result[1]
            
            not_rob_this = max(left_result) + max(right_result)
            
            return [rob_this, not_rob_this]
        
        result = rob_helper(root)
        return max(result) if result else 0
    
    # Test with sample tree
    def create_sample_tree():
        """Create sample tree: [3,2,3,null,3,null,1]"""
        root = TreeNode(3)
        root.left = TreeNode(2)
        root.right = TreeNode(3)
        root.left.right = TreeNode(3)
        root.right.right = TreeNode(1)
        return root
    
    sample_tree = create_sample_tree()
    
    print("House Robber Tree Variants:")
    print("=" * 50)
    
    basic_rob = rob_tree_dp_optimal(sample_tree)
    print(f"Basic robbing: {basic_rob}")
    
    constrained_rob = rob_with_constraints(sample_tree, 2)
    print(f"Rob at most 2 houses: {constrained_rob}")
    
    costs = [1, 2, 3]  # Increasing costs by level
    cost_rob = rob_with_costs(sample_tree, costs)
    print(f"With level costs {costs}: {cost_rob}")
    
    guards = {"L"}  # Guard at left child of root
    guard_rob = rob_tree_with_guards(sample_tree, guards)
    print(f"With guards at {guards}: {guard_rob}")


# Test cases
def test_house_robber_tree():
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
    
    def tree_to_list(root):
        """Convert tree back to list for display"""
        if not root:
            return []
        
        result = []
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)
        
        # Remove trailing None values
        while result and result[-1] is None:
            result.pop()
        
        return result
    
    test_cases = [
        ([3, 2, 3, None, 3, None, 1], 7),
        ([3, 4, 5, 1, 3, None, 1], 9),
        ([1], 1),
        ([2, 1, 3, None, 4], 7),
        ([4, 1, None, 2, None, 3], 7),
        ([5, 4, 9, 1, None, None, 2], 15),
        ([], 0)
    ]
    
    print("Testing House Robber III Solutions:")
    print("=" * 70)
    
    for i, (tree_list, expected) in enumerate(test_cases):
        print(f"\nTest Case {i + 1}:")
        print(f"Tree: {tree_list}")
        print(f"Expected: {expected}")
        
        root = create_tree_from_list(tree_list)
        
        # Skip brute force for larger trees
        if len([x for x in tree_list if x is not None]) <= 7:
            try:
                brute_force = rob_recursive_brute_force(root)
                print(f"Brute Force:      {brute_force:>4} {'✓' if brute_force == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        memoization = rob_memoization(root)
        tree_dp = rob_tree_dp_optimal(root)
        iterative = rob_iterative_postorder(root)
        
        print(f"Memoization:      {memoization:>4} {'✓' if memoization == expected else '✗'}")
        print(f"Tree DP:          {tree_dp:>4} {'✓' if tree_dp == expected else '✗'}")
        print(f"Iterative:        {iterative:>4} {'✓' if iterative == expected else '✗'}")
        
        # Show detailed analysis for interesting cases
        if len([x for x in tree_list if x is not None]) > 0 and len([x for x in tree_list if x is not None]) <= 8:
            max_rob, optimal_nodes = rob_tree_dp_detailed(root)
            print(f"Optimal houses to rob: {optimal_nodes}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    sample_tree = create_tree_from_list([3, 2, 3, None, 3, None, 1])
    rob_tree_analysis(sample_tree)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    rob_tree_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. STATE DECOMPOSITION: Each node has two states (rob/not rob)")
    print("2. CONSTRAINT PROPAGATION: Robbing a node affects children")
    print("3. OPTIMAL SUBSTRUCTURE: Optimal solution contains optimal subsolutions")
    print("4. TREE TRAVERSAL: Post-order ensures children processed before parent")
    print("5. SPACE OPTIMIZATION: No memoization needed with proper state design")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Resource Allocation: Optimal selection with adjacency constraints")
    print("• Network Security: Optimal placement with conflict constraints")
    print("• Game Theory: Territory control with adjacency rules")
    print("• Scheduling: Task selection with dependency constraints")
    print("• Graph Theory: Maximum independent set in trees")


if __name__ == "__main__":
    test_house_robber_tree()


"""
HOUSE ROBBER III - FUNDAMENTAL TREE DP PATTERN:
===============================================

This problem establishes the core Tree DP framework:
- State propagation from children to parent
- Binary choice at each node (rob vs. not rob)
- Constraint handling (adjacent nodes cannot both be robbed)
- Optimal substructure in tree context

KEY INSIGHTS:
============
1. **BINARY STATE**: Each node has exactly two states (rob/not rob)
2. **CONSTRAINT PROPAGATION**: Parent choice affects valid child choices
3. **BOTTOM-UP PROCESSING**: Children must be solved before parent
4. **OPTIMAL SUBSTRUCTURE**: Global optimum contains local optima
5. **SPACE EFFICIENCY**: No memoization needed with proper state design

ALGORITHM APPROACHES:
====================

1. **Brute Force Recursive**: O(2^n) time, O(h) space
   - Try both choices at each node
   - Massive overlapping subproblems

2. **Memoization**: O(n) time, O(n) space
   - Cache results by node reference
   - Still requires extra memory

3. **Optimal Tree DP**: O(n) time, O(h) space
   - Return two values per node: [rob_this, not_rob_this]
   - Most elegant and efficient solution

4. **Iterative Post-order**: O(n) time, O(h) space
   - Stack-based traversal with state tracking
   - Avoids recursion completely

CORE TREE DP ALGORITHM:
======================
```python
def rob(root):
    def helper(node):
        if not node:
            return [0, 0]  # [rob_this, not_rob_this]
        
        left = helper(node.left)
        right = helper(node.right)
        
        rob_this = node.val + left[1] + right[1]      # Can't rob children
        not_rob_this = max(left) + max(right)         # Can rob children optimally
        
        return [rob_this, not_rob_this]
    
    return max(helper(root))
```

STATE TRANSITION LOGIC:
======================
**For each node, we track two states**:
- `rob_this`: Maximum money if we rob this house
- `not_rob_this`: Maximum money if we don't rob this house

**Recurrence Relations**:
```
rob_this[node] = node.val + not_rob_this[left] + not_rob_this[right]
not_rob_this[node] = max(rob_this[left], not_rob_this[left]) + 
                     max(rob_this[right], not_rob_this[right])
```

**Base Case**: `rob_helper(null) = [0, 0]`

CONSTRAINT HANDLING:
===================
**Adjacent Constraint**: If we rob a house, we cannot rob its directly connected neighbors.

**Implementation**: When robbing current node, we can only take the "not robbed" values from children.

**Flexibility**: When not robbing current node, we can take the optimal choice from children.

TREE TRAVERSAL PATTERN:
=======================
**Post-order Requirement**: Must process children before parent
- Children's optimal values needed to compute parent's optimal values
- Natural fit for recursive tree DP

**Processing Order**:
1. Recursively solve left subtree
2. Recursively solve right subtree  
3. Combine results to solve current node
4. Return optimal values for current subtree

COMPLEXITY ANALYSIS:
===================
- **Time**: O(n) - each node visited exactly once
- **Space**: O(h) - recursion stack depth equals tree height
- **Best Case Space**: O(log n) for balanced tree
- **Worst Case Space**: O(n) for completely unbalanced tree

SPACE OPTIMIZATION:
==================
**No Memoization Needed**: Unlike traditional DP problems, Tree DP doesn't need memoization table because:
- Each node is visited exactly once in post-order traversal
- No overlapping subproblems in tree structure
- State flows naturally from children to parent

**Memory Efficiency**: Only requires O(h) space for recursion stack vs O(n) for memoization table.

APPLICATIONS:
============
- **Resource Allocation**: Optimal selection with adjacency constraints
- **Network Security**: Optimal sensor placement with interference constraints
- **Game Theory**: Territory control with conflict rules
- **Facility Location**: Optimal placement with competition constraints
- **Graph Theory**: Maximum independent set in trees (polynomial time solution)

RELATED PROBLEMS:
================
- **Binary Tree Maximum Path Sum (124)**: Similar tree traversal with state management
- **Distribute Coins in Binary Tree (979)**: Resource flow optimization
- **Binary Tree Cameras (968)**: Coverage optimization with constraints
- **Maximum Independent Set**: General graph version (NP-hard vs polynomial for trees)

VARIANTS:
========
- **Limited Resources**: Rob at most k houses
- **Weighted Costs**: Different costs for robbing different levels
- **Guards**: Some houses cannot be robbed
- **Multiple Thieves**: Coordination between multiple actors

MATHEMATICAL PROPERTIES:
========================
- **Optimal Substructure**: Optimal solution contains optimal subsolutions
- **Greedy Choice Property**: Sometimes optimal to make locally optimal choices
- **Overlapping Subproblems**: Not present in tree structure (unlike general graphs)
- **Monotonicity**: More flexibility (larger subtrees) never decreases optimal value

This problem beautifully demonstrates the power of Tree DP:
combining tree traversal with dynamic programming to solve
optimization problems with structural constraints efficiently.
"""
