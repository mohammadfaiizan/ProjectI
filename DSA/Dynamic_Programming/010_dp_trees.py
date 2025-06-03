"""
Dynamic Programming - Tree Patterns
This module implements various DP problems on trees including house robber,
maximum path sum, diameter calculations, and node-dependent recursion with memoization.
"""

from typing import List, Dict, Tuple, Optional, Any
import time
from collections import defaultdict, deque

# ==================== TREE NODE DEFINITION ====================

class TreeNode:
    """Binary tree node definition"""
    def __init__(self, val: int = 0, left: 'Optional[TreeNode]' = None, right: 'Optional[TreeNode]' = None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

# ==================== HOUSE ROBBER ON TREES ====================

class HouseRobberTree:
    """
    House Robber III Problems
    
    LeetCode 337 - House Robber III
    Rob houses arranged in binary tree without robbing adjacent nodes.
    """
    
    def rob_tree_recursive(self, root: Optional[TreeNode]) -> int:
        """
        Recursive solution with memoization
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            root: Root of binary tree
        
        Returns:
            Maximum money that can be robbed
        """
        memo = {}
        
        def rob_helper(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            if node in memo:
                return memo[node]
            
            # Option 1: Rob this house (skip children, rob grandchildren)
            rob_current = node.val
            if node.left:
                rob_current += rob_helper(node.left.left) + rob_helper(node.left.right)
            if node.right:
                rob_current += rob_helper(node.right.left) + rob_helper(node.right.right)
            
            # Option 2: Don't rob this house (rob children)
            skip_current = rob_helper(node.left) + rob_helper(node.right)
            
            result = max(rob_current, skip_current)
            memo[node] = result
            return result
        
        return rob_helper(root)
    
    def rob_tree_optimized(self, root: Optional[TreeNode]) -> int:
        """
        Optimized solution returning both states
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        
        Returns tuple: (max_rob_excluding_current, max_rob_including_current)
        """
        def rob_helper(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                return (0, 0)  # (rob_excluding, rob_including)
            
            left_exclude, left_include = rob_helper(node.left)
            right_exclude, right_include = rob_helper(node.right)
            
            # If we rob current node, we can't rob its children
            rob_current = node.val + left_exclude + right_exclude
            
            # If we don't rob current, take max from children
            skip_current = max(left_exclude, left_include) + max(right_exclude, right_include)
            
            return (skip_current, rob_current)
        
        exclude, include = rob_helper(root)
        return max(exclude, include)
    
    def rob_tree_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Find maximum money and the actual nodes robbed
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (max_money, list_of_robbed_values)
        """
        def rob_helper(node: Optional[TreeNode]) -> Tuple[Tuple[int, List[int]], Tuple[int, List[int]]]:
            if not node:
                return ((0, []), (0, []))
            
            left_result = rob_helper(node.left)
            right_result = rob_helper(node.right)
            
            # Don't rob current
            exclude_money = (left_result[0][0] + left_result[1][0], 
                           right_result[0][0] + right_result[1][0])
            exclude_path = left_result[0][1] + left_result[1][1] + right_result[0][1] + right_result[1][1]
            
            if exclude_money[0] >= exclude_money[1]:
                exclude_state = (exclude_money[0], left_result[0][1] + right_result[0][1])
            else:
                exclude_state = (exclude_money[1], left_result[1][1] + right_result[1][1])
            
            # Rob current
            include_money = node.val + left_result[0][0] + right_result[0][0]
            include_path = [node.val] + left_result[0][1] + right_result[0][1]
            include_state = (include_money, include_path)
            
            return (exclude_state, include_state)
        
        exclude_result, include_result = rob_helper(root)
        
        if exclude_result[0] >= include_result[0]:
            return exclude_result
        else:
            return include_result
    
    def rob_tree_k_distance(self, root: Optional[TreeNode], k: int) -> int:
        """
        House robber with k-distance constraint
        Cannot rob houses within distance k of each other
        
        Args:
            root: Root of binary tree
            k: Minimum distance between robbed houses
        """
        memo = {}
        
        def get_distance_k_nodes(node: Optional[TreeNode], distance: int) -> List[TreeNode]:
            """Get all nodes at exactly distance k from current node"""
            if not node or distance < 0:
                return []
            
            if distance == 0:
                return [node]
            
            result = []
            if node.left:
                result.extend(get_distance_k_nodes(node.left, distance - 1))
            if node.right:
                result.extend(get_distance_k_nodes(node.right, distance - 1))
            
            return result
        
        def rob_helper(node: Optional[TreeNode], robbed_ancestors: set) -> int:
            if not node:
                return 0
            
            # Check if we can rob this node (not within k distance of robbed ancestors)
            can_rob = node not in robbed_ancestors
            
            # Option 1: Don't rob this node
            skip_current = rob_helper(node.left, robbed_ancestors) + rob_helper(node.right, robbed_ancestors)
            
            if not can_rob:
                return skip_current
            
            # Option 2: Rob this node
            new_robbed = robbed_ancestors | {node}
            # Add nodes within distance k
            for dist in range(1, k + 1):
                distance_k_nodes = get_distance_k_nodes(node, dist)
                new_robbed.update(distance_k_nodes)
            
            rob_current = node.val + rob_helper(node.left, new_robbed) + rob_helper(node.right, new_robbed)
            
            return max(skip_current, rob_current)
        
        return rob_helper(root, set())

# ==================== MAXIMUM PATH SUM PROBLEMS ====================

class MaximumPathSum:
    """
    Maximum Path Sum Problems in Binary Trees
    
    Various problems involving finding paths with maximum sum
    in binary trees with different constraints.
    """
    
    def max_path_sum_any_to_any(self, root: Optional[TreeNode]) -> int:
        """
        Maximum path sum between any two nodes
        
        LeetCode 124 - Binary Tree Maximum Path Sum
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            Maximum path sum
        """
        max_sum = float('-inf')
        
        def max_path_helper(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            
            if not node:
                return 0
            
            # Get maximum path sum starting from left and right children
            left_max = max(0, max_path_helper(node.left))   # Ignore negative paths
            right_max = max(0, max_path_helper(node.right))
            
            # Maximum path sum through current node (as turning point)
            current_max = node.val + left_max + right_max
            max_sum = max(max_sum, current_max)
            
            # Return maximum path sum starting from current node
            return node.val + max(left_max, right_max)
        
        max_path_helper(root)
        return max_sum
    
    def max_path_sum_root_to_leaf(self, root: Optional[TreeNode]) -> int:
        """
        Maximum path sum from root to any leaf
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        """
        def max_path_helper(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            
            if not node.left and not node.right:  # Leaf node
                return node.val
            
            left_max = float('-inf')
            right_max = float('-inf')
            
            if node.left:
                left_max = max_path_helper(node.left)
            if node.right:
                right_max = max_path_helper(node.right)
            
            return node.val + max(left_max, right_max)
        
        return max_path_helper(root) if root else 0
    
    def max_path_sum_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Find maximum path sum and the actual path
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (max_sum, path_values)
        """
        max_sum = float('-inf')
        best_path = []
        
        def max_path_helper(node: Optional[TreeNode]) -> Tuple[int, List[int]]:
            nonlocal max_sum, best_path
            
            if not node:
                return 0, []
            
            left_sum, left_path = max_path_helper(node.left)
            right_sum, right_path = max_path_helper(node.right)
            
            # Choose better path to extend
            if left_sum >= right_sum and left_sum > 0:
                current_sum = node.val + left_sum
                current_path = [node.val] + left_path
            elif right_sum > 0:
                current_sum = node.val + right_sum
                current_path = [node.val] + right_path
            else:
                current_sum = node.val
                current_path = [node.val]
            
            # Check if path through this node is better
            through_node_sum = node.val + max(0, left_sum) + max(0, right_sum)
            through_node_path = (left_path[::-1] if left_sum > 0 else []) + [node.val] + (right_path if right_sum > 0 else [])
            
            if through_node_sum > max_sum:
                max_sum = through_node_sum
                best_path = through_node_path
            
            return current_sum, current_path
        
        max_path_helper(root)
        return max_sum, best_path
    
    def max_path_sum_with_k_nodes(self, root: Optional[TreeNode], k: int) -> int:
        """
        Maximum path sum with exactly k nodes
        
        Args:
            root: Root of binary tree
            k: Exact number of nodes in path
        """
        memo = {}
        
        def max_path_helper(node: Optional[TreeNode], remaining_nodes: int, can_turn: bool) -> int:
            if not node or remaining_nodes <= 0:
                return 0 if remaining_nodes == 0 else float('-inf')
            
            if (node, remaining_nodes, can_turn) in memo:
                return memo[(node, remaining_nodes, can_turn)]
            
            if remaining_nodes == 1:
                result = node.val
            else:
                result = float('-inf')
                
                # Continue in one direction (no turning)
                if node.left:
                    result = max(result, node.val + max_path_helper(node.left, remaining_nodes - 1, False))
                if node.right:
                    result = max(result, node.val + max_path_helper(node.right, remaining_nodes - 1, False))
                
                # Turn at this node (use both children)
                if can_turn and node.left and node.right:
                    for left_nodes in range(1, remaining_nodes):
                        right_nodes = remaining_nodes - left_nodes - 1
                        if right_nodes > 0:
                            path_sum = (node.val + 
                                      max_path_helper(node.left, left_nodes, False) +
                                      max_path_helper(node.right, right_nodes, False))
                            result = max(result, path_sum)
            
            memo[(node, remaining_nodes, can_turn)] = result
            return result
        
        return max_path_helper(root, k, True)

# ==================== DIAMETER PROBLEMS ====================

class TreeDiameter:
    """
    Tree Diameter Problems
    
    Find diameter (longest path) in trees using DP optimization.
    """
    
    def diameter_binary_tree(self, root: Optional[TreeNode]) -> int:
        """
        Diameter of binary tree
        
        LeetCode 543 - Diameter of Binary Tree
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            Length of diameter (number of edges)
        """
        max_diameter = 0
        
        def depth_helper(node: Optional[TreeNode]) -> int:
            nonlocal max_diameter
            
            if not node:
                return 0
            
            left_depth = depth_helper(node.left)
            right_depth = depth_helper(node.right)
            
            # Diameter through current node
            current_diameter = left_depth + right_depth
            max_diameter = max(max_diameter, current_diameter)
            
            # Return depth of subtree rooted at current node
            return 1 + max(left_depth, right_depth)
        
        depth_helper(root)
        return max_diameter
    
    def diameter_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Find diameter and the actual path
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (diameter_length, path_nodes)
        """
        max_diameter = 0
        diameter_path = []
        
        def depth_helper(node: Optional[TreeNode]) -> Tuple[int, List[int]]:
            nonlocal max_diameter, diameter_path
            
            if not node:
                return 0, []
            
            left_depth, left_path = depth_helper(node.left)
            right_depth, right_path = depth_helper(node.right)
            
            # Diameter through current node
            current_diameter = left_depth + right_depth
            
            if current_diameter > max_diameter:
                max_diameter = current_diameter
                # Construct path: left_path (reversed) + current + right_path
                diameter_path = left_path[::-1] + [node.val] + right_path
            
            # Return depth and path of deeper subtree
            if left_depth >= right_depth:
                return 1 + left_depth, [node.val] + left_path
            else:
                return 1 + right_depth, [node.val] + right_path
        
        depth_helper(root)
        return max_diameter, diameter_path
    
    def diameter_n_ary_tree(self, root: 'Node') -> int:
        """
        Diameter of N-ary tree
        
        Args:
            root: Root of N-ary tree
        
        Returns:
            Diameter length
        """
        max_diameter = 0
        
        def depth_helper(node) -> int:
            nonlocal max_diameter
            
            if not node:
                return 0
            
            # Get depths of all children
            child_depths = []
            for child in node.children:
                child_depths.append(depth_helper(child))
            
            child_depths.sort(reverse=True)
            
            # Diameter through current node is sum of two largest depths
            if len(child_depths) >= 2:
                current_diameter = child_depths[0] + child_depths[1]
            elif len(child_depths) == 1:
                current_diameter = child_depths[0]
            else:
                current_diameter = 0
            
            max_diameter = max(max_diameter, current_diameter)
            
            # Return depth of subtree rooted at current node
            return 1 + (child_depths[0] if child_depths else 0)
        
        depth_helper(root)
        return max_diameter
    
    def weighted_diameter(self, root: Optional[TreeNode]) -> int:
        """
        Diameter of tree with weighted edges
        Assumes each edge has weight equal to the minimum of its endpoints
        """
        max_diameter = 0
        
        def depth_helper(node: Optional[TreeNode]) -> int:
            nonlocal max_diameter
            
            if not node:
                return 0
            
            left_depth = depth_helper(node.left)
            right_depth = depth_helper(node.right)
            
            # Calculate weighted depths
            left_weighted = left_depth + min(node.val, node.left.val) if node.left else 0
            right_weighted = right_depth + min(node.val, node.right.val) if node.right else 0
            
            # Diameter through current node
            current_diameter = left_weighted + right_weighted
            max_diameter = max(max_diameter, current_diameter)
            
            # Return maximum weighted depth
            return max(left_weighted, right_weighted)
        
        depth_helper(root)
        return max_diameter

# ==================== NODE-DEPENDENT RECURSION ====================

class NodeDependentRecursion:
    """
    Advanced tree DP with node-dependent states
    
    Complex tree traversal problems where the state depends on
    node properties and multiple recursive calls.
    """
    
    def max_path_sum_alternating_signs(self, root: Optional[TreeNode]) -> int:
        """
        Maximum path sum where we must alternate between positive and negative nodes
        
        Args:
            root: Root of binary tree
        
        Returns:
            Maximum alternating path sum
        """
        memo = {}
        
        def dfs(node: Optional[TreeNode], need_positive: bool) -> int:
            if not node:
                return 0
            
            if (node, need_positive) in memo:
                return memo[(node, need_positive)]
            
            # Check if current node satisfies the requirement
            current_positive = node.val > 0
            
            if need_positive != current_positive:
                # Node doesn't satisfy requirement, skip it
                result = max(
                    dfs(node.left, need_positive),
                    dfs(node.right, need_positive),
                    0  # Option to not take any path
                )
            else:
                # Node satisfies requirement, we can use it
                result = node.val
                
                # Try extending path to children (flip requirement)
                left_sum = dfs(node.left, not need_positive)
                right_sum = dfs(node.right, not need_positive)
                
                result = max(
                    result,  # Just current node
                    result + left_sum,   # Current + left
                    result + right_sum,  # Current + right
                    result + left_sum + right_sum  # Current + both (if valid path)
                )
            
            memo[(node, need_positive)] = result
            return result
        
        return max(dfs(root, True), dfs(root, False))
    
    def count_good_nodes(self, root: Optional[TreeNode]) -> int:
        """
        Count good nodes (nodes >= all ancestors in path from root)
        
        LeetCode 1448 - Count Good Nodes in Binary Tree
        
        Args:
            root: Root of binary tree
        
        Returns:
            Number of good nodes
        """
        def dfs(node: Optional[TreeNode], max_so_far: int) -> int:
            if not node:
                return 0
            
            # Current node is good if it's >= max value in path from root
            is_good = 1 if node.val >= max_so_far else 0
            
            # Update max for children
            new_max = max(max_so_far, node.val)
            
            return is_good + dfs(node.left, new_max) + dfs(node.right, new_max)
        
        return dfs(root, float('-inf'))
    
    def max_path_sum_with_node_types(self, root: Optional[TreeNode], 
                                   node_types: Dict[TreeNode, str]) -> int:
        """
        Maximum path sum where nodes have types and certain transitions are allowed
        
        Args:
            root: Root of binary tree
            node_types: Dictionary mapping nodes to their types
        
        Returns:
            Maximum valid path sum
        """
        # Define valid transitions
        valid_transitions = {
            'A': ['B', 'C'],
            'B': ['A', 'C'],
            'C': ['A', 'B']
        }
        
        memo = {}
        
        def dfs(node: Optional[TreeNode], parent_type: Optional[str]) -> int:
            if not node:
                return 0
            
            if (node, parent_type) in memo:
                return memo[(node, parent_type)]
            
            current_type = node_types.get(node, 'A')  # Default type
            
            # Check if transition from parent to current is valid
            if parent_type and current_type not in valid_transitions.get(parent_type, []):
                # Invalid transition, skip this node
                result = max(
                    dfs(node.left, parent_type),
                    dfs(node.right, parent_type),
                    0
                )
            else:
                # Valid transition, can include this node
                result = node.val
                
                # Try extending to children
                left_sum = dfs(node.left, current_type)
                right_sum = dfs(node.right, current_type)
                
                result = max(
                    result,
                    result + left_sum,
                    result + right_sum,
                    result + left_sum + right_sum
                )
            
            memo[(node, parent_type)] = result
            return result
        
        return dfs(root, None)
    
    def tree_coloring_dp(self, root: Optional[TreeNode], colors: int) -> int:
        """
        Count ways to color tree such that no adjacent nodes have same color
        
        Args:
            root: Root of binary tree
            colors: Number of available colors
        
        Returns:
            Number of valid colorings
        """
        memo = {}
        
        def dfs(node: Optional[TreeNode], parent_color: int) -> int:
            if not node:
                return 1
            
            if (node, parent_color) in memo:
                return memo[(node, parent_color)]
            
            total_ways = 0
            
            # Try each color for current node
            for color in range(colors):
                if color != parent_color:  # Can't use parent's color
                    # Count ways to color children
                    left_ways = dfs(node.left, color)
                    right_ways = dfs(node.right, color)
                    total_ways += left_ways * right_ways
            
            memo[(node, parent_color)] = total_ways
            return total_ways
        
        return dfs(root, -1)  # Root has no parent color constraint

# ==================== TREE UTILITY FUNCTIONS ====================

def build_tree_from_array(arr: List[Optional[int]]) -> Optional[TreeNode]:
    """Build binary tree from level-order array representation"""
    if not arr or arr[0] is None:
        return None
    
    root = TreeNode(arr[0])
    queue = deque([root])
    i = 1
    
    while queue and i < len(arr):
        node = queue.popleft()
        
        # Left child
        if i < len(arr) and arr[i] is not None:
            node.left = TreeNode(arr[i])
            queue.append(node.left)
        i += 1
        
        # Right child
        if i < len(arr) and arr[i] is not None:
            node.right = TreeNode(arr[i])
            queue.append(node.right)
        i += 1
    
    return root

def print_tree_inorder(root: Optional[TreeNode]) -> List[int]:
    """Print tree in inorder traversal"""
    result = []
    
    def inorder(node):
        if node:
            inorder(node.left)
            result.append(node.val)
            inorder(node.right)
    
    inorder(root)
    return result

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different tree DP approaches"""
    print("=== Tree DP Performance Analysis ===\n")
    
    import random
    
    # Build test trees of different sizes
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        # Create balanced tree with random values
        values = [random.randint(-100, 100) for _ in range(size)]
        root = build_tree_from_array(values)
        
        print(f"Tree with {size} nodes:")
        
        # House Robber
        robber = HouseRobberTree()
        
        start_time = time.time()
        rob_recursive = robber.rob_tree_recursive(root)
        time_recursive = time.time() - start_time
        
        start_time = time.time()
        rob_optimized = robber.rob_tree_optimized(root)
        time_optimized = time.time() - start_time
        
        print(f"  House Robber:")
        print(f"    Recursive + Memo: {rob_recursive} ({time_recursive:.6f}s)")
        print(f"    Optimized: {rob_optimized} ({time_optimized:.6f}s)")
        print(f"    Results match: {rob_recursive == rob_optimized}")
        
        # Maximum Path Sum
        path_sum = MaximumPathSum()
        
        start_time = time.time()
        max_path = path_sum.max_path_sum_any_to_any(root)
        time_path = time.time() - start_time
        
        print(f"  Max Path Sum: {max_path} ({time_path:.6f}s)")
        
        # Diameter
        diameter = TreeDiameter()
        
        start_time = time.time()
        tree_diameter = diameter.diameter_binary_tree(root)
        time_diameter = time.time() - start_time
        
        print(f"  Diameter: {tree_diameter} ({time_diameter:.6f}s)")
        print()

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Tree DP Demo ===\n")
    
    # Build test tree: [3, 2, 3, null, 3, null, 1]
    test_tree = build_tree_from_array([3, 2, 3, None, 3, None, 1])
    
    print("Test tree (inorder):", print_tree_inorder(test_tree))
    print()
    
    # House Robber Problems
    print("1. House Robber on Trees:")
    robber = HouseRobberTree()
    
    max_rob_recursive = robber.rob_tree_recursive(test_tree)
    max_rob_optimized = robber.rob_tree_optimized(test_tree)
    max_rob_with_path, robbed_nodes = robber.rob_tree_with_path(test_tree)
    
    print(f"  Maximum money (recursive): {max_rob_recursive}")
    print(f"  Maximum money (optimized): {max_rob_optimized}")
    print(f"  Maximum money with path: {max_rob_with_path}")
    print(f"  Robbed nodes: {robbed_nodes}")
    
    # K-distance constraint
    k_distance_rob = robber.rob_tree_k_distance(test_tree, 2)
    print(f"  Max money with 2-distance constraint: {k_distance_rob}")
    print()
    
    # Maximum Path Sum Problems
    print("2. Maximum Path Sum Problems:")
    path_sum = MaximumPathSum()
    
    # Build tree for path sum: [1, 2, 3]
    path_tree = build_tree_from_array([1, 2, 3])
    
    max_any_to_any = path_sum.max_path_sum_any_to_any(path_tree)
    max_root_to_leaf = path_sum.max_path_sum_root_to_leaf(path_tree)
    max_with_path, best_path = path_sum.max_path_sum_with_path(path_tree)
    
    print(f"  Tree: {print_tree_inorder(path_tree)}")
    print(f"  Max path sum (any to any): {max_any_to_any}")
    print(f"  Max path sum (root to leaf): {max_root_to_leaf}")
    print(f"  Max path with actual path: {max_with_path}, path: {best_path}")
    
    # K nodes constraint
    max_k_nodes = path_sum.max_path_sum_with_k_nodes(path_tree, 2)
    print(f"  Max path sum with exactly 2 nodes: {max_k_nodes}")
    print()
    
    # Diameter Problems
    print("3. Tree Diameter Problems:")
    diameter = TreeDiameter()
    
    # Build tree for diameter: [1, 2, 3, 4, 5]
    diameter_tree = build_tree_from_array([1, 2, 3, 4, 5])
    
    tree_diameter = diameter.diameter_binary_tree(diameter_tree)
    diameter_with_path, diameter_path = diameter.diameter_with_path(diameter_tree)
    
    print(f"  Tree: {print_tree_inorder(diameter_tree)}")
    print(f"  Diameter: {tree_diameter}")
    print(f"  Diameter with path: {diameter_with_path}, path: {diameter_path}")
    
    # Weighted diameter
    weighted_diam = diameter.weighted_diameter(diameter_tree)
    print(f"  Weighted diameter: {weighted_diam}")
    print()
    
    # Node-Dependent Recursion
    print("4. Node-Dependent Recursion:")
    node_dependent = NodeDependentRecursion()
    
    # Build tree: [3, 1, 4, 3, null, 1, 5]
    complex_tree = build_tree_from_array([3, 1, 4, 3, None, 1, 5])
    
    alternating_sum = node_dependent.max_path_sum_alternating_signs(complex_tree)
    good_nodes = node_dependent.count_good_nodes(complex_tree)
    
    print(f"  Tree: {print_tree_inorder(complex_tree)}")
    print(f"  Max alternating path sum: {alternating_sum}")
    print(f"  Count of good nodes: {good_nodes}")
    
    # Tree coloring
    coloring_ways = node_dependent.tree_coloring_dp(complex_tree, 3)
    print(f"  Ways to color tree with 3 colors: {coloring_ways}")
    
    # Node types example
    node_types = {}
    for node in [complex_tree, complex_tree.left, complex_tree.right]:
        if node:
            node_types[node] = 'A' if node.val % 2 == 1 else 'B'
    
    typed_path_sum = node_dependent.max_path_sum_with_node_types(complex_tree, node_types)
    print(f"  Max path sum with node types: {typed_path_sum}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("=== Tree DP Pattern Recognition ===")
    print("Common Tree DP Patterns:")
    print("  1. Bottom-up: Process children before parent")
    print("  2. State tracking: Return multiple values (include/exclude, depths, etc.)")
    print("  3. Path problems: Consider paths ending vs passing through nodes")
    print("  4. Memoization: Cache results for nodes with same state")
    print("  5. Multiple traversals: Sometimes need multiple passes")
    
    print("\nKey Techniques:")
    print("  1. Return tuples: (max_including_current, max_excluding_current)")
    print("  2. Global variables: For tracking global optimum (diameter, max path)")
    print("  3. State compression: Use node properties as state")
    print("  4. Ancestor tracking: Pass information down from parent")
    
    print("\nProblem Classification:")
    print("  1. Single node decisions: Rob/don't rob, include/exclude")
    print("  2. Path optimization: Maximum/minimum path sums")
    print("  3. Structural properties: Diameter, height, balance")
    print("  4. Coloring/labeling: Assignment problems with constraints")
    print("  5. Counting: Number of valid configurations")
    
    print("\nReal-world Applications:")
    print("  1. Network optimization (router placement)")
    print("  2. Decision trees and game theory")
    print("  3. Phylogenetic analysis in biology")
    print("  4. Compiler optimization (expression trees)")
    print("  5. Social network analysis")
    print("  6. File system optimization")
    
    print("\nCommon Pitfalls:")
    print("  1. Forgetting base cases (null nodes)")
    print("  2. Not handling single node trees")
    print("  3. Incorrect state transitions")
    print("  4. Memory issues with large trees")
    print("  5. Not considering all path types (ending vs passing through)")
    
    print("\n=== Demo Complete ===") 