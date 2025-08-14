"""
Tree Basics - Fundamental Tree Concepts and Operations
This module covers tree definitions, types, basic traversals, and construction methods.
"""

from collections import deque
from typing import List, Optional, Tuple, Dict

class TreeNode:
    """Basic tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeBasics:
    
    def __init__(self):
        """Initialize tree basics with utility methods"""
        pass
    
    # ==================== TREE DEFINITIONS AND PROPERTIES ====================
    
    def get_height(self, root: Optional[TreeNode]) -> int:
        """
        Get height of tree (longest path from root to leaf)
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        
        Args:
            root: Root of the tree
        
        Returns:
            int: Height of tree (-1 for empty tree)
        """
        if not root:
            return -1
        
        return 1 + max(self.get_height(root.left), self.get_height(root.right))
    
    def get_depth(self, root: Optional[TreeNode], target_node: TreeNode, depth: int = 0) -> int:
        """
        Get depth of a specific node (distance from root)
        
        Args:
            root: Root of the tree
            target_node: Node to find depth for
            depth: Current depth (used in recursion)
        
        Returns:
            int: Depth of target node, -1 if not found
        """
        if not root:
            return -1
        
        if root == target_node:
            return depth
        
        # Search in left subtree
        left_depth = self.get_depth(root.left, target_node, depth + 1)
        if left_depth != -1:
            return left_depth
        
        # Search in right subtree
        return self.get_depth(root.right, target_node, depth + 1)
    
    def count_nodes(self, root: Optional[TreeNode]) -> int:
        """Count total number of nodes in tree"""
        if not root:
            return 0
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)
    
    def count_leaf_nodes(self, root: Optional[TreeNode]) -> int:
        """Count number of leaf nodes"""
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        return self.count_leaf_nodes(root.left) + self.count_leaf_nodes(root.right)
    
    def count_internal_nodes(self, root: Optional[TreeNode]) -> int:
        """Count number of internal (non-leaf) nodes"""
        if not root or (not root.left and not root.right):
            return 0
        return 1 + self.count_internal_nodes(root.left) + self.count_internal_nodes(root.right)
    
    # ==================== TREE TYPE CLASSIFICATIONS ====================
    
    def is_full_binary_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is full (every node has 0 or 2 children)
        
        Args:
            root: Root of the tree
        
        Returns:
            bool: True if tree is full
        """
        if not root:
            return True
        
        # If leaf node
        if not root.left and not root.right:
            return True
        
        # If both children exist
        if root.left and root.right:
            return (self.is_full_binary_tree(root.left) and 
                   self.is_full_binary_tree(root.right))
        
        # If only one child exists
        return False
    
    def is_complete_binary_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is complete (all levels filled except possibly last, left-filled)
        
        Args:
            root: Root of the tree
        
        Returns:
            bool: True if tree is complete
        """
        if not root:
            return True
        
        queue = deque([root])
        flag = False
        
        while queue:
            node = queue.popleft()
            
            # Check left child
            if node.left:
                if flag:  # If we've seen a node with missing child
                    return False
                queue.append(node.left)
            else:
                flag = True
            
            # Check right child
            if node.right:
                if flag:  # If we've seen a node with missing child
                    return False
                queue.append(node.right)
            else:
                flag = True
        
        return True
    
    def is_perfect_binary_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is perfect (all internal nodes have 2 children, all leaves at same level)
        
        Args:
            root: Root of the tree
        
        Returns:
            bool: True if tree is perfect
        """
        if not root:
            return True
        
        height = self.get_height(root)
        return self._is_perfect_helper(root, height, 0)
    
    def _is_perfect_helper(self, root: Optional[TreeNode], height: int, level: int) -> bool:
        """Helper function for perfect tree check"""
        if not root:
            return True
        
        # If leaf node, check if it's at the correct level
        if not root.left and not root.right:
            return level == height
        
        # If only one child, not perfect
        if not root.left or not root.right:
            return False
        
        # Recursively check both subtrees
        return (self._is_perfect_helper(root.left, height, level + 1) and
                self._is_perfect_helper(root.right, height, level + 1))
    
    def is_balanced_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is balanced (height difference between subtrees <= 1)
        
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
    
    # ==================== BASIC TREE TRAVERSALS ====================
    
    def inorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Inorder traversal: Left -> Root -> Right
        
        Time Complexity: O(n)
        Space Complexity: O(h) for recursion stack
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Inorder traversal result
        """
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    def preorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Preorder traversal: Root -> Left -> Right
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Preorder traversal result
        """
        result = []
        
        def preorder(node):
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        return result
    
    def postorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Postorder traversal: Left -> Right -> Root
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Postorder traversal result
        """
        result = []
        
        def postorder(node):
            if node:
                postorder(node.left)
                postorder(node.right)
                result.append(node.val)
        
        postorder(root)
        return result
    
    def level_order_traversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Level order traversal (BFS): Level by level
        
        Time Complexity: O(n)
        Space Complexity: O(w) where w is maximum width
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Level order traversal result (list of levels)
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level_nodes = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level_nodes.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level_nodes)
        
        return result
    
    def level_order_flat(self, root: Optional[TreeNode]) -> List[int]:
        """Level order traversal returning flat list"""
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    # ==================== TREE CONSTRUCTION ====================
    
    def build_tree_from_array(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
        """
        Build binary tree from array representation (level order)
        None represents missing nodes
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            arr: Array representation of tree
        
        Returns:
            TreeNode: Root of constructed tree
        """
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
    
    def tree_to_array(self, root: Optional[TreeNode]) -> List[Optional[int]]:
        """
        Convert tree to array representation
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Array representation of tree
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
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
    
    def build_tree_from_inorder_preorder(self, inorder: List[int], preorder: List[int]) -> Optional[TreeNode]:
        """
        Build binary tree from inorder and preorder traversals
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            inorder: Inorder traversal
            preorder: Preorder traversal
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not inorder or not preorder:
            return None
        
        # Create a map for quick lookup of indices in inorder
        inorder_map = {val: i for i, val in enumerate(inorder)}
        self.preorder_idx = 0
        
        def build_tree(left, right):
            if left > right:
                return None
            
            # Root is the current element in preorder
            root_val = preorder[self.preorder_idx]
            root = TreeNode(root_val)
            self.preorder_idx += 1
            
            # Find root position in inorder
            root_idx = inorder_map[root_val]
            
            # Build left subtree first (preorder: root, left, right)
            root.left = build_tree(left, root_idx - 1)
            root.right = build_tree(root_idx + 1, right)
            
            return root
        
        return build_tree(0, len(inorder) - 1)
    
    def build_tree_from_inorder_postorder(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """
        Build binary tree from inorder and postorder traversals
        
        Args:
            inorder: Inorder traversal
            postorder: Postorder traversal
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not inorder or not postorder:
            return None
        
        inorder_map = {val: i for i, val in enumerate(inorder)}
        self.postorder_idx = len(postorder) - 1
        
        def build_tree(left, right):
            if left > right:
                return None
            
            # Root is the current element in postorder (from right to left)
            root_val = postorder[self.postorder_idx]
            root = TreeNode(root_val)
            self.postorder_idx -= 1
            
            # Find root position in inorder
            root_idx = inorder_map[root_val]
            
            # Build right subtree first (postorder: left, right, root)
            root.right = build_tree(root_idx + 1, right)
            root.left = build_tree(left, root_idx - 1)
            
            return root
        
        return build_tree(0, len(inorder) - 1)
    
    def build_tree_from_string(self, data: str) -> Optional[TreeNode]:
        """
        Build tree from string representation like "1(2(4)(5))(3(6)(7))"
        
        Args:
            data: String representation of tree
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not data:
            return None
        
        def build_tree_helper(index):
            if index[0] >= len(data):
                return None
            
            # Read the number
            start = index[0]
            if data[index[0]] == '-':
                index[0] += 1
            
            while index[0] < len(data) and data[index[0]].isdigit():
                index[0] += 1
            
            val = int(data[start:index[0]])
            root = TreeNode(val)
            
            # Check for left child
            if index[0] < len(data) and data[index[0]] == '(':
                index[0] += 1  # Skip '('
                root.left = build_tree_helper(index)
                index[0] += 1  # Skip ')'
            
            # Check for right child
            if index[0] < len(data) and data[index[0]] == '(':
                index[0] += 1  # Skip '('
                root.right = build_tree_helper(index)
                index[0] += 1  # Skip ')'
            
            return root
        
        index = [0]
        return build_tree_helper(index)
    
    # ==================== UTILITY METHODS ====================
    
    def print_tree_structure(self, root: Optional[TreeNode], level: int = 0, prefix: str = "Root: "):
        """Print tree structure in a readable format"""
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
    
    def get_tree_statistics(self, root: Optional[TreeNode]) -> Dict[str, int]:
        """Get comprehensive statistics about the tree"""
        if not root:
            return {
                "total_nodes": 0,
                "leaf_nodes": 0,
                "internal_nodes": 0,
                "height": -1,
                "is_full": True,
                "is_complete": True,
                "is_perfect": True,
                "is_balanced": True
            }
        
        return {
            "total_nodes": self.count_nodes(root),
            "leaf_nodes": self.count_leaf_nodes(root),
            "internal_nodes": self.count_internal_nodes(root),
            "height": self.get_height(root),
            "is_full": self.is_full_binary_tree(root),
            "is_complete": self.is_complete_binary_tree(root),
            "is_perfect": self.is_perfect_binary_tree(root),
            "is_balanced": self.is_balanced_tree(root)
        }
    
    def are_trees_equal(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """Check if two trees are structurally identical"""
        if not root1 and not root2:
            return True
        
        if not root1 or not root2:
            return False
        
        return (root1.val == root2.val and
                self.are_trees_equal(root1.left, root2.left) and
                self.are_trees_equal(root1.right, root2.right))


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Tree Basics Demo ===\n")
    
    tree_basics = TreeBasics()
    
    # Example 1: Build tree from array and analyze
    print("1. Tree Construction and Analysis:")
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    root = tree_basics.build_tree_from_array(arr)
    
    print(f"Original array: {arr}")
    print("Tree structure:")
    tree_basics.print_tree_structure(root)
    
    stats = tree_basics.get_tree_statistics(root)
    print(f"Tree statistics: {stats}")
    print()
    
    # Example 2: Tree traversals
    print("2. Tree Traversals:")
    print(f"Inorder:    {tree_basics.inorder_traversal(root)}")
    print(f"Preorder:   {tree_basics.preorder_traversal(root)}")
    print(f"Postorder:  {tree_basics.postorder_traversal(root)}")
    print(f"Level order: {tree_basics.level_order_traversal(root)}")
    print(f"Level flat:  {tree_basics.level_order_flat(root)}")
    print()
    
    # Example 3: Tree type classifications
    print("3. Tree Type Classifications:")
    
    # Perfect binary tree
    perfect_arr = [1, 2, 3, 4, 5, 6, 7]
    perfect_root = tree_basics.build_tree_from_array(perfect_arr)
    perfect_stats = tree_basics.get_tree_statistics(perfect_root)
    print(f"Perfect tree {perfect_arr}:")
    print(f"  Full: {perfect_stats['is_full']}")
    print(f"  Complete: {perfect_stats['is_complete']}")
    print(f"  Perfect: {perfect_stats['is_perfect']}")
    print(f"  Balanced: {perfect_stats['is_balanced']}")
    
    # Incomplete tree
    incomplete_arr = [1, 2, 3, 4, None, 6, 7]
    incomplete_root = tree_basics.build_tree_from_array(incomplete_arr)
    incomplete_stats = tree_basics.get_tree_statistics(incomplete_root)
    print(f"Incomplete tree {incomplete_arr}:")
    print(f"  Full: {incomplete_stats['is_full']}")
    print(f"  Complete: {incomplete_stats['is_complete']}")
    print(f"  Perfect: {incomplete_stats['is_perfect']}")
    print(f"  Balanced: {incomplete_stats['is_balanced']}")
    print()
    
    # Example 4: Tree construction from traversals
    print("4. Tree Construction from Traversals:")
    inorder = [4, 2, 5, 1, 6, 3, 7]
    preorder = [1, 2, 4, 5, 3, 6, 7]
    postorder = [4, 5, 2, 6, 7, 3, 1]
    
    print(f"Inorder:  {inorder}")
    print(f"Preorder: {preorder}")
    print(f"Postorder: {postorder}")
    
    # Build from inorder + preorder
    tree_from_pre = tree_basics.build_tree_from_inorder_preorder(inorder, preorder)
    print("Tree from inorder + preorder:")
    tree_basics.print_tree_structure(tree_from_pre)
    
    # Build from inorder + postorder
    tree_from_post = tree_basics.build_tree_from_inorder_postorder(inorder, postorder)
    print("Tree from inorder + postorder:")
    tree_basics.print_tree_structure(tree_from_post)
    
    # Verify they're the same
    are_equal = tree_basics.are_trees_equal(tree_from_pre, tree_from_post)
    print(f"Both constructions are equal: {are_equal}")
    print()
    
    # Example 5: String representation
    print("5. Tree from String Representation:")
    tree_string = "1(2(4)(5))(3(6)(7))"
    print(f"String representation: {tree_string}")
    
    tree_from_string = tree_basics.build_tree_from_string(tree_string)
    print("Constructed tree:")
    tree_basics.print_tree_structure(tree_from_string)
    
    # Convert back to array
    arr_from_tree = tree_basics.tree_to_array(tree_from_string)
    print(f"Array representation: {arr_from_tree}")
    
    print("\n=== Demo Complete ===") 