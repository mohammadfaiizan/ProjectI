"""
Binary Tree Problems - Common Binary Tree Algorithms and Problems
This module implements essential binary tree problems and algorithms.
"""

from collections import deque
from typing import List, Optional, Tuple

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class BinaryTreeProblems:
    
    def __init__(self):
        """Initialize binary tree problems solver"""
        pass
    
    # ==================== TREE DIAMETER AND HEIGHT ====================
    
    def diameter_of_tree(self, root: Optional[TreeNode]) -> int:
        """
        Find diameter of binary tree (longest path between any two nodes)
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        
        Args:
            root: Root of the tree
        
        Returns:
            int: Diameter of the tree
        """
        self.diameter = 0
        
        def height(node):
            if not node:
                return 0
            
            left_height = height(node.left)
            right_height = height(node.right)
            
            # Update diameter if path through current node is longer
            self.diameter = max(self.diameter, left_height + right_height)
            
            return 1 + max(left_height, right_height)
        
        height(root)
        return self.diameter
    
    def height_of_tree(self, root: Optional[TreeNode]) -> int:
        """
        Find height/depth of binary tree
        
        Args:
            root: Root of the tree
        
        Returns:
            int: Height of tree (-1 for empty tree)
        """
        if not root:
            return -1
        
        return 1 + max(self.height_of_tree(root.left), 
                      self.height_of_tree(root.right))
    
    def height_iterative(self, root: Optional[TreeNode]) -> int:
        """Find height using level order traversal (iterative)"""
        if not root:
            return -1
        
        queue = deque([root])
        height = -1
        
        while queue:
            height += 1
            level_size = len(queue)
            
            for _ in range(level_size):
                node = queue.popleft()
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return height
    
    # ==================== BALANCED TREE OPERATIONS ====================
    
    def is_balanced(self, root: Optional[TreeNode]) -> bool:
        """
        Check if binary tree is height-balanced
        
        Args:
            root: Root of the tree
        
        Returns:
            bool: True if tree is balanced
        """
        def check_balance(node):
            if not node:
                return True, 0
            
            left_balanced, left_height = check_balance(node.left)
            if not left_balanced:
                return False, 0
            
            right_balanced, right_height = check_balance(node.right)
            if not right_balanced:
                return False, 0
            
            # Check if current node is balanced
            if abs(left_height - right_height) > 1:
                return False, 0
            
            return True, 1 + max(left_height, right_height)
        
        balanced, _ = check_balance(root)
        return balanced
    
    def balance_bst(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """Convert unbalanced BST to balanced BST"""
        # Get inorder traversal (sorted array)
        inorder = []
        
        def get_inorder(node):
            if node:
                get_inorder(node.left)
                inorder.append(node.val)
                get_inorder(node.right)
        
        get_inorder(root)
        
        # Build balanced BST from sorted array
        def build_balanced(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            node = TreeNode(inorder[mid])
            
            node.left = build_balanced(left, mid - 1)
            node.right = build_balanced(mid + 1, right)
            
            return node
        
        return build_balanced(0, len(inorder) - 1)
    
    # ==================== TREE INVERSION AND MIRRORING ====================
    
    def invert_tree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Invert/flip binary tree (mirror image)
        
        Args:
            root: Root of the tree
        
        Returns:
            TreeNode: Root of inverted tree
        """
        if not root:
            return None
        
        # Swap left and right children
        root.left, root.right = root.right, root.left
        
        # Recursively invert subtrees
        self.invert_tree(root.left)
        self.invert_tree(root.right)
        
        return root
    
    def invert_tree_iterative(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """Iterative tree inversion using queue"""
        if not root:
            return None
        
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
            # Swap children
            node.left, node.right = node.right, node.left
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return root
    
    def is_mirror(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """Check if two trees are mirror images"""
        if not root1 and not root2:
            return True
        
        if not root1 or not root2:
            return False
        
        return (root1.val == root2.val and
                self.is_mirror(root1.left, root2.right) and
                self.is_mirror(root1.right, root2.left))
    
    def is_symmetric(self, root: Optional[TreeNode]) -> bool:
        """Check if tree is symmetric (mirror of itself)"""
        if not root:
            return True
        
        return self.is_mirror(root.left, root.right)
    
    # ==================== SUM TREE OPERATIONS ====================
    
    def is_sum_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is a sum tree (each node = sum of subtrees)
        
        Args:
            root: Root of the tree
        
        Returns:
            bool: True if tree is sum tree
        """
        def check_sum_tree(node):
            if not node:
                return True, 0
            
            if not node.left and not node.right:
                return True, node.val
            
            left_is_sum, left_sum = check_sum_tree(node.left)
            right_is_sum, right_sum = check_sum_tree(node.right)
            
            if not left_is_sum or not right_is_sum:
                return False, 0
            
            if node.val == left_sum + right_sum:
                return True, node.val + left_sum + right_sum
            else:
                return False, 0
        
        is_sum, _ = check_sum_tree(root)
        return is_sum
    
    def convert_to_sum_tree(self, root: Optional[TreeNode]) -> int:
        """
        Convert tree to sum tree and return sum of original tree
        
        Args:
            root: Root of the tree
        
        Returns:
            int: Sum of original tree values
        """
        if not root:
            return 0
        
        old_val = root.val
        
        left_sum = self.convert_to_sum_tree(root.left)
        right_sum = self.convert_to_sum_tree(root.right)
        
        root.val = left_sum + right_sum
        
        return old_val + left_sum + right_sum
    
    def has_children_sum_property(self, root: Optional[TreeNode]) -> bool:
        """Check if tree has children sum property"""
        if not root or (not root.left and not root.right):
            return True
        
        children_sum = 0
        if root.left:
            children_sum += root.left.val
        if root.right:
            children_sum += root.right.val
        
        return (root.val == children_sum and
                self.has_children_sum_property(root.left) and
                self.has_children_sum_property(root.right))
    
    # ==================== MAXIMUM PATH SUM ====================
    
    def max_path_sum(self, root: Optional[TreeNode]) -> int:
        """
        Find maximum path sum in binary tree (path can start/end anywhere)
        
        Args:
            root: Root of the tree
        
        Returns:
            int: Maximum path sum
        """
        self.max_sum = float('-inf')
        
        def max_gain(node):
            if not node:
                return 0
            
            # Maximum gain from left and right subtrees
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # Current path sum including current node
            current_path_sum = node.val + left_gain + right_gain
            
            # Update global maximum
            self.max_sum = max(self.max_sum, current_path_sum)
            
            # Return maximum gain if we continue path from current node
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum
    
    def max_path_sum_leaf_to_leaf(self, root: Optional[TreeNode]) -> int:
        """Maximum path sum from leaf to leaf"""
        self.max_sum = float('-inf')
        
        def max_path_helper(node):
            if not node:
                return 0
            
            if not node.left and not node.right:
                return node.val
            
            left_sum = float('-inf')
            right_sum = float('-inf')
            
            if node.left:
                left_sum = max_path_helper(node.left)
            if node.right:
                right_sum = max_path_helper(node.right)
            
            # If both children exist
            if node.left and node.right:
                self.max_sum = max(self.max_sum, left_sum + right_sum + node.val)
                return node.val + max(left_sum, right_sum)
            
            # If only one child exists
            return node.val + max(left_sum, right_sum)
        
        max_path_helper(root)
        return self.max_sum
    
    # ==================== ROOT TO LEAF PATH PROBLEMS ====================
    
    def has_path_sum(self, root: Optional[TreeNode], target_sum: int) -> bool:
        """
        Check if there exists root-to-leaf path with given sum
        
        Args:
            root: Root of the tree
            target_sum: Target sum
        
        Returns:
            bool: True if path exists
        """
        if not root:
            return False
        
        # If leaf node, check if remaining sum equals node value
        if not root.left and not root.right:
            return target_sum == root.val
        
        # Check left and right subtrees with reduced sum
        remaining_sum = target_sum - root.val
        return (self.has_path_sum(root.left, remaining_sum) or
                self.has_path_sum(root.right, remaining_sum))
    
    def find_path_sum(self, root: Optional[TreeNode], target_sum: int) -> List[int]:
        """Find a root-to-leaf path with given sum"""
        def dfs(node, current_sum, path):
            if not node:
                return False
            
            path.append(node.val)
            current_sum += node.val
            
            # If leaf node and sum matches
            if not node.left and not node.right and current_sum == target_sum:
                return True
            
            # Try left and right subtrees
            if (dfs(node.left, current_sum, path) or
                dfs(node.right, current_sum, path)):
                return True
            
            # Backtrack
            path.pop()
            return False
        
        path = []
        if dfs(root, 0, path):
            return path
        return []
    
    def find_all_path_sums(self, root: Optional[TreeNode], target_sum: int) -> List[List[int]]:
        """Find all root-to-leaf paths with given sum"""
        result = []
        
        def dfs(node, current_sum, path):
            if not node:
                return
            
            path.append(node.val)
            current_sum += node.val
            
            # If leaf node and sum matches
            if not node.left and not node.right and current_sum == target_sum:
                result.append(path[:])  # Make a copy
            else:
                # Continue to children
                dfs(node.left, current_sum, path)
                dfs(node.right, current_sum, path)
            
            # Backtrack
            path.pop()
        
        dfs(root, 0, [])
        return result
    
    def print_all_root_to_leaf_paths(self, root: Optional[TreeNode]) -> List[List[int]]:
        """Print all root-to-leaf paths"""
        result = []
        
        def dfs(node, path):
            if not node:
                return
            
            path.append(node.val)
            
            # If leaf node
            if not node.left and not node.right:
                result.append(path[:])  # Make a copy
            else:
                dfs(node.left, path)
                dfs(node.right, path)
            
            # Backtrack
            path.pop()
        
        dfs(root, [])
        return result
    
    # ==================== PATH COUNT PROBLEMS ====================
    
    def count_paths_with_sum(self, root: Optional[TreeNode], target_sum: int) -> int:
        """
        Count paths with given sum (can start/end at any node)
        
        Args:
            root: Root of the tree
            target_sum: Target sum
        
        Returns:
            int: Number of paths with target sum
        """
        def count_paths_from_node(node, target):
            if not node:
                return 0
            
            count = 0
            if node.val == target:
                count += 1
            
            count += count_paths_from_node(node.left, target - node.val)
            count += count_paths_from_node(node.right, target - node.val)
            
            return count
        
        if not root:
            return 0
        
        # Paths starting from root
        paths_from_root = count_paths_from_node(root, target_sum)
        
        # Paths in left and right subtrees
        paths_in_left = self.count_paths_with_sum(root.left, target_sum)
        paths_in_right = self.count_paths_with_sum(root.right, target_sum)
        
        return paths_from_root + paths_in_left + paths_in_right
    
    def count_paths_with_sum_optimized(self, root: Optional[TreeNode], target_sum: int) -> int:
        """Optimized version using prefix sum"""
        def dfs(node, current_sum, prefix_sums):
            if not node:
                return 0
            
            current_sum += node.val
            count = prefix_sums.get(current_sum - target_sum, 0)
            
            # Add current sum to prefix sums
            prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1
            
            # Recurse to children
            count += dfs(node.left, current_sum, prefix_sums)
            count += dfs(node.right, current_sum, prefix_sums)
            
            # Remove current sum from prefix sums (backtrack)
            prefix_sums[current_sum] -= 1
            
            return count
        
        return dfs(root, 0, {0: 1})
    
    # ==================== TREE COMPARISON AND UTILITIES ====================
    
    def are_identical(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """Check if two trees are identical"""
        if not root1 and not root2:
            return True
        
        if not root1 or not root2:
            return False
        
        return (root1.val == root2.val and
                self.are_identical(root1.left, root2.left) and
                self.are_identical(root1.right, root2.right))
    
    def is_subtree(self, main_tree: Optional[TreeNode], subtree: Optional[TreeNode]) -> bool:
        """Check if subtree is a subtree of main tree"""
        if not subtree:
            return True
        
        if not main_tree:
            return False
        
        return (self.are_identical(main_tree, subtree) or
                self.is_subtree(main_tree.left, subtree) or
                self.is_subtree(main_tree.right, subtree))
    
    def count_nodes(self, root: Optional[TreeNode]) -> int:
        """Count total number of nodes"""
        if not root:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)
    
    def count_leaf_nodes(self, root: Optional[TreeNode]) -> int:
        """Count leaf nodes"""
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        return self.count_leaf_nodes(root.left) + self.count_leaf_nodes(root.right)
    
    def sum_of_all_nodes(self, root: Optional[TreeNode]) -> int:
        """Calculate sum of all nodes"""
        if not root:
            return 0
        return root.val + self.sum_of_all_nodes(root.left) + self.sum_of_all_nodes(root.right)
    
    # ==================== UTILITY METHODS ====================
    
    def build_tree_from_array(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
        """Build tree from array representation"""
        if not arr or arr[0] is None:
            return None
        
        root = TreeNode(arr[0])
        queue = deque([root])
        i = 1
        
        while queue and i < len(arr):
            node = queue.popleft()
            
            if i < len(arr) and arr[i] is not None:
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
            
            if i < len(arr) and arr[i] is not None:
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    def print_tree_structure(self, root: Optional[TreeNode], level: int = 0, prefix: str = "Root: "):
        """Print tree structure"""
        if root is not None:
            print(" " * (level * 4) + prefix + str(root.val))
            if root.left is not None or root.right is not None:
                if root.left:
                    self.print_tree_structure(root.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if root.right:
                    self.print_tree_structure(root.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Binary Tree Problems Demo ===\n")
    
    problems = BinaryTreeProblems()
    
    # Example 1: Tree diameter and height
    print("1. Tree Diameter and Height:")
    arr1 = [1, 2, 3, 4, 5, None, 6, 7, 8]
    root1 = problems.build_tree_from_array(arr1)
    
    print("Tree structure:")
    problems.print_tree_structure(root1)
    
    diameter = problems.diameter_of_tree(root1)
    height = problems.height_of_tree(root1)
    height_iter = problems.height_iterative(root1)
    
    print(f"Diameter: {diameter}")
    print(f"Height (recursive): {height}")
    print(f"Height (iterative): {height_iter}")
    print()
    
    # Example 2: Balanced tree operations
    print("2. Balanced Tree Operations:")
    is_balanced = problems.is_balanced(root1)
    print(f"Is balanced: {is_balanced}")
    
    # Create unbalanced tree
    unbalanced_arr = [1, 2, None, 3, None, None, None, 4]
    unbalanced_root = problems.build_tree_from_array(unbalanced_arr)
    print("Unbalanced tree:")
    problems.print_tree_structure(unbalanced_root)
    
    is_unbalanced_balanced = problems.is_balanced(unbalanced_root)
    print(f"Unbalanced tree is balanced: {is_unbalanced_balanced}")
    print()
    
    # Example 3: Tree inversion
    print("3. Tree Inversion:")
    arr2 = [4, 2, 7, 1, 3, 6, 9]
    root2 = problems.build_tree_from_array(arr2)
    
    print("Original tree:")
    problems.print_tree_structure(root2)
    
    inverted = problems.invert_tree(root2)
    print("Inverted tree:")
    problems.print_tree_structure(inverted)
    print()
    
    # Example 4: Sum tree operations
    print("4. Sum Tree Operations:")
    sum_tree_arr = [26, 10, 3, 4, 6, None, 3]
    sum_tree_root = problems.build_tree_from_array(sum_tree_arr)
    
    print("Tree for sum operations:")
    problems.print_tree_structure(sum_tree_root)
    
    is_sum_tree = problems.is_sum_tree(sum_tree_root)
    print(f"Is sum tree: {is_sum_tree}")
    
    has_children_sum = problems.has_children_sum_property(sum_tree_root)
    print(f"Has children sum property: {has_children_sum}")
    print()
    
    # Example 5: Maximum path sum
    print("5. Maximum Path Sum:")
    path_sum_arr = [-10, 9, 20, None, None, 15, 7]
    path_sum_root = problems.build_tree_from_array(path_sum_arr)
    
    print("Tree for path sum:")
    problems.print_tree_structure(path_sum_root)
    
    max_path = problems.max_path_sum(path_sum_root)
    print(f"Maximum path sum: {max_path}")
    print()
    
    # Example 6: Root to leaf path problems
    print("6. Root to Leaf Path Problems:")
    path_arr = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, None, 1]
    path_root = problems.build_tree_from_array(path_arr)
    
    print("Tree for path problems:")
    problems.print_tree_structure(path_root)
    
    target_sum = 22
    has_path = problems.has_path_sum(path_root, target_sum)
    print(f"Has path with sum {target_sum}: {has_path}")
    
    path_with_sum = problems.find_path_sum(path_root, target_sum)
    print(f"Path with sum {target_sum}: {path_with_sum}")
    
    all_paths_with_sum = problems.find_all_path_sums(path_root, target_sum)
    print(f"All paths with sum {target_sum}: {all_paths_with_sum}")
    
    all_paths = problems.print_all_root_to_leaf_paths(path_root)
    print(f"All root-to-leaf paths: {all_paths}")
    print()
    
    # Example 7: Path count problems
    print("7. Path Count Problems:")
    count_arr = [10, 5, -3, 3, 2, None, 11, 3, -2, None, 1]
    count_root = problems.build_tree_from_array(count_arr)
    
    print("Tree for path counting:")
    problems.print_tree_structure(count_root)
    
    target_sum_count = 8
    path_count = problems.count_paths_with_sum(count_root, target_sum_count)
    path_count_opt = problems.count_paths_with_sum_optimized(count_root, target_sum_count)
    
    print(f"Paths with sum {target_sum_count}: {path_count}")
    print(f"Paths with sum {target_sum_count} (optimized): {path_count_opt}")
    print()
    
    # Example 8: Tree comparison
    print("8. Tree Comparison:")
    tree1 = problems.build_tree_from_array([1, 2, 3])
    tree2 = problems.build_tree_from_array([1, 2, 3])
    tree3 = problems.build_tree_from_array([1, 2, 4])
    
    identical_1_2 = problems.are_identical(tree1, tree2)
    identical_1_3 = problems.are_identical(tree1, tree3)
    
    print(f"Tree1 and Tree2 are identical: {identical_1_2}")
    print(f"Tree1 and Tree3 are identical: {identical_1_3}")
    
    subtree = problems.build_tree_from_array([2])
    is_sub = problems.is_subtree(tree1, subtree)
    print(f"[2] is subtree of [1,2,3]: {is_sub}")
    print()
    
    # Example 9: Tree statistics
    print("9. Tree Statistics:")
    stats_root = problems.build_tree_from_array([1, 2, 3, 4, 5, 6, 7])
    
    total_nodes = problems.count_nodes(stats_root)
    leaf_nodes = problems.count_leaf_nodes(stats_root)
    total_sum = problems.sum_of_all_nodes(stats_root)
    
    print(f"Total nodes: {total_nodes}")
    print(f"Leaf nodes: {leaf_nodes}")
    print(f"Sum of all nodes: {total_sum}")
    
    print("\n=== Demo Complete ===") 