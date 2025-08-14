"""
Tree Recursion - Binary Trees and Tree Algorithms
=================================================

Topics: Tree traversals, tree construction, tree properties, tree modifications
Companies: Google, Amazon, Microsoft, Facebook, Apple, LinkedIn
Difficulty: Medium to Hard
Time Complexity: O(n) for most tree operations
Space Complexity: O(h) where h is height of tree
"""

from typing import List, Optional, Tuple, Any
from collections import deque

class TreeNode:
    """Binary tree node definition"""
    def __init__(self, val: int = 0, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeRecursion:
    
    def __init__(self):
        """Initialize with call tracking"""
        self.call_count = 0
        self.max_depth = 0
        self.current_depth = 0
    
    # ==========================================
    # 1. TREE TRAVERSALS
    # ==========================================
    
    def inorder_traversal(self, root: Optional[TreeNode], result: List[int] = None) -> List[int]:
        """
        Inorder traversal: Left → Root → Right
        
        For BST, this gives sorted order
        Time: O(n), Space: O(h)
        """
        if result is None:
            result = []
        
        self.call_count += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        if root:
            print(f"{'  ' * (self.current_depth-1)}Visiting node {root.val} (inorder)")
            
            # Traverse left subtree
            self.inorder_traversal(root.left, result)
            
            # Process current node
            result.append(root.val)
            print(f"{'  ' * (self.current_depth-1)}Processing node {root.val}")
            
            # Traverse right subtree
            self.inorder_traversal(root.right, result)
        
        self.current_depth -= 1
        return result
    
    def preorder_traversal(self, root: Optional[TreeNode], result: List[int] = None) -> List[int]:
        """
        Preorder traversal: Root → Left → Right
        
        Useful for copying/serializing trees
        Time: O(n), Space: O(h)
        """
        if result is None:
            result = []
        
        self.call_count += 1
        
        if root:
            # Process current node first
            result.append(root.val)
            print(f"Processing node {root.val} (preorder)")
            
            # Then traverse left and right subtrees
            self.preorder_traversal(root.left, result)
            self.preorder_traversal(root.right, result)
        
        return result
    
    def postorder_traversal(self, root: Optional[TreeNode], result: List[int] = None) -> List[int]:
        """
        Postorder traversal: Left → Right → Root
        
        Useful for deleting/calculating tree properties
        Time: O(n), Space: O(h)
        """
        if result is None:
            result = []
        
        self.call_count += 1
        
        if root:
            # Traverse children first
            self.postorder_traversal(root.left, result)
            self.postorder_traversal(root.right, result)
            
            # Process current node last
            result.append(root.val)
            print(f"Processing node {root.val} (postorder)")
        
        return result
    
    def level_order_recursive(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Level order traversal using recursion
        
        Time: O(n), Space: O(h + w) where w is max width
        """
        result = []
        
        def traverse_level(node: Optional[TreeNode], level: int) -> None:
            if not node:
                return
            
            # Add new level if needed
            if level >= len(result):
                result.append([])
            
            # Add current node to its level
            result[level].append(node.val)
            
            # Traverse children at next level
            traverse_level(node.left, level + 1)
            traverse_level(node.right, level + 1)
        
        traverse_level(root, 0)
        return result
    
    # ==========================================
    # 2. TREE PROPERTIES
    # ==========================================
    
    def tree_height(self, root: Optional[TreeNode]) -> int:
        """
        Calculate height of tree
        
        Height = longest path from root to leaf
        Time: O(n), Space: O(h)
        """
        self.call_count += 1
        
        if not root:
            return 0
        
        left_height = self.tree_height(root.left)
        right_height = self.tree_height(root.right)
        
        height = 1 + max(left_height, right_height)
        print(f"Height of subtree rooted at {root.val}: {height}")
        
        return height
    
    def tree_size(self, root: Optional[TreeNode]) -> int:
        """
        Count total number of nodes in tree
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        left_size = self.tree_size(root.left)
        right_size = self.tree_size(root.right)
        
        total_size = 1 + left_size + right_size
        print(f"Size of subtree rooted at {root.val}: {total_size}")
        
        return total_size
    
    def tree_sum(self, root: Optional[TreeNode]) -> int:
        """
        Calculate sum of all nodes in tree
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        left_sum = self.tree_sum(root.left)
        right_sum = self.tree_sum(root.right)
        
        total_sum = root.val + left_sum + right_sum
        print(f"Sum of subtree rooted at {root.val}: {total_sum}")
        
        return total_sum
    
    def is_balanced(self, root: Optional[TreeNode]) -> Tuple[bool, int]:
        """
        Check if tree is height-balanced
        
        A tree is balanced if height difference between left and right
        subtrees is at most 1 for every node
        
        Time: O(n), Space: O(h)
        Returns: (is_balanced, height)
        """
        if not root:
            return True, 0
        
        # Check left and right subtrees
        left_balanced, left_height = self.is_balanced(root.left)
        right_balanced, right_height = self.is_balanced(root.right)
        
        # Current node is balanced if:
        # 1. Both subtrees are balanced
        # 2. Height difference is at most 1
        current_balanced = (left_balanced and 
                          right_balanced and 
                          abs(left_height - right_height) <= 1)
        
        current_height = 1 + max(left_height, right_height)
        
        print(f"Node {root.val}: balanced={current_balanced}, height={current_height}")
        
        return current_balanced, current_height
    
    def is_symmetric(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is symmetric (mirror of itself)
        
        Time: O(n), Space: O(h)
        """
        def is_mirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            # Both null
            if not left and not right:
                return True
            
            # One null, other not
            if not left or not right:
                return False
            
            # Both non-null: check value and recursive mirror property
            return (left.val == right.val and 
                   is_mirror(left.left, right.right) and 
                   is_mirror(left.right, right.left))
        
        if not root:
            return True
        
        return is_mirror(root.left, root.right)
    
    # ==========================================
    # 3. TREE CONSTRUCTION
    # ==========================================
    
    def build_tree_from_preorder_inorder(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        Construct tree from preorder and inorder traversals
        
        Time: O(n), Space: O(n)
        """
        if not preorder or not inorder:
            return None
        
        # First element in preorder is always root
        root_val = preorder[0]
        root = TreeNode(root_val)
        
        print(f"Creating node {root_val}")
        
        # Find root position in inorder
        root_index = inorder.index(root_val)
        
        # Split inorder into left and right subtrees
        left_inorder = inorder[:root_index]
        right_inorder = inorder[root_index + 1:]
        
        # Split preorder accordingly
        left_preorder = preorder[1:1 + len(left_inorder)]
        right_preorder = preorder[1 + len(left_inorder):]
        
        print(f"  Left subtree - preorder: {left_preorder}, inorder: {left_inorder}")
        print(f"  Right subtree - preorder: {right_preorder}, inorder: {right_inorder}")
        
        # Recursively build left and right subtrees
        root.left = self.build_tree_from_preorder_inorder(left_preorder, left_inorder)
        root.right = self.build_tree_from_preorder_inorder(right_preorder, right_inorder)
        
        return root
    
    def build_tree_from_array(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
        """
        Build tree from level-order array representation
        
        None represents missing nodes
        Time: O(n), Space: O(n)
        """
        if not arr or arr[0] is None:
            return None
        
        root = TreeNode(arr[0])
        queue = [root]
        i = 1
        
        while queue and i < len(arr):
            node = queue.pop(0)
            
            # Add left child
            if i < len(arr) and arr[i] is not None:
                node.left = TreeNode(arr[i])
                queue.append(node.left)
            i += 1
            
            # Add right child
            if i < len(arr) and arr[i] is not None:
                node.right = TreeNode(arr[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    # ==========================================
    # 4. TREE MODIFICATIONS
    # ==========================================
    
    def invert_tree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Invert binary tree (mirror it)
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return None
        
        print(f"Inverting subtree rooted at {root.val}")
        
        # Swap left and right children
        root.left, root.right = root.right, root.left
        
        # Recursively invert subtrees
        self.invert_tree(root.left)
        self.invert_tree(root.right)
        
        return root
    
    def flatten_tree_to_linked_list(self, root: Optional[TreeNode]) -> None:
        """
        Flatten tree to linked list in-place (preorder)
        
        Time: O(n), Space: O(h)
        """
        def flatten_helper(node: Optional[TreeNode]) -> Optional[TreeNode]:
            """Returns tail of flattened subtree"""
            if not node:
                return None
            
            # Flatten left and right subtrees
            left_tail = flatten_helper(node.left)
            right_tail = flatten_helper(node.right)
            
            # If left subtree exists, insert it between node and right subtree
            if left_tail:
                left_tail.right = node.right
                node.right = node.left
                node.left = None
            
            # Return tail of current subtree
            return right_tail or left_tail or node
        
        flatten_helper(root)
    
    # ==========================================
    # 5. TREE SEARCH AND PATHS
    # ==========================================
    
    def find_path_to_node(self, root: Optional[TreeNode], target: int) -> List[int]:
        """
        Find path from root to target node
        
        Time: O(n), Space: O(h)
        """
        path = []
        
        def find_path_helper(node: Optional[TreeNode]) -> bool:
            if not node:
                return False
            
            # Add current node to path
            path.append(node.val)
            print(f"Visiting node {node.val}, path: {path}")
            
            # Check if current node is target
            if node.val == target:
                print(f"Found target {target}!")
                return True
            
            # Search in left and right subtrees
            if (find_path_helper(node.left) or 
                find_path_helper(node.right)):
                return True
            
            # Backtrack: remove current node from path
            path.pop()
            print(f"Backtracking from node {node.val}, path: {path}")
            return False
        
        if find_path_helper(root):
            return path
        return []
    
    def all_root_to_leaf_paths(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Find all paths from root to leaves
        
        Time: O(n * h), Space: O(n * h)
        """
        result = []
        
        def find_paths(node: Optional[TreeNode], current_path: List[int]) -> None:
            if not node:
                return
            
            # Add current node to path
            current_path.append(node.val)
            
            # If leaf node, add path to result
            if not node.left and not node.right:
                result.append(current_path[:])  # Make a copy
                print(f"Found leaf path: {current_path}")
            else:
                # Continue searching in children
                find_paths(node.left, current_path)
                find_paths(node.right, current_path)
            
            # Backtrack
            current_path.pop()
        
        find_paths(root, [])
        return result
    
    def path_sum(self, root: Optional[TreeNode], target_sum: int) -> bool:
        """
        Check if there exists a root-to-leaf path with given sum
        
        Time: O(n), Space: O(h)
        """
        def has_path_sum(node: Optional[TreeNode], remaining_sum: int) -> bool:
            if not node:
                return False
            
            # Update remaining sum
            remaining_sum -= node.val
            print(f"Node {node.val}, remaining sum: {remaining_sum}")
            
            # If leaf node, check if sum matches
            if not node.left and not node.right:
                return remaining_sum == 0
            
            # Check left or right subtree
            return (has_path_sum(node.left, remaining_sum) or 
                   has_path_sum(node.right, remaining_sum))
        
        return has_path_sum(root, target_sum)
    
    # ==========================================
    # 6. LOWEST COMMON ANCESTOR
    # ==========================================
    
    def lowest_common_ancestor(self, root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
        """
        Find lowest common ancestor of two nodes
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return None
        
        print(f"Checking node {root.val} for LCA of {p.val} and {q.val}")
        
        # If current node is one of the target nodes
        if root == p or root == q:
            print(f"Found target node {root.val}")
            return root
        
        # Search in left and right subtrees
        left_lca = self.lowest_common_ancestor(root.left, p, q)
        right_lca = self.lowest_common_ancestor(root.right, p, q)
        
        # If both subtrees contain target nodes, current node is LCA
        if left_lca and right_lca:
            print(f"Node {root.val} is LCA")
            return root
        
        # Return non-null result
        return left_lca or right_lca
    
    # ==========================================
    # 7. TREE UTILITIES
    # ==========================================
    
    def print_tree_structure(self, root: Optional[TreeNode], level: int = 0, prefix: str = "Root: ") -> None:
        """Pretty print tree structure"""
        if root:
            print(" " * (level * 4) + prefix + str(root.val))
            if root.left or root.right:
                if root.left:
                    self.print_tree_structure(root.left, level + 1, "L--- ")
                else:
                    print(" " * ((level + 1) * 4) + "L--- None")
                
                if root.right:
                    self.print_tree_structure(root.right, level + 1, "R--- ")
                else:
                    print(" " * ((level + 1) * 4) + "R--- None")

# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_tree_recursion():
    """Demonstrate all tree recursion concepts"""
    print("=== TREE RECURSION DEMONSTRATION ===\n")
    
    tr = TreeRecursion()
    
    # Create sample tree:
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    print("Sample tree structure:")
    tr.print_tree_structure(root)
    print()
    
    # 1. Tree Traversals
    print("=== TREE TRAVERSALS ===")
    tr.call_count = 0
    tr.max_depth = 0
    tr.current_depth = 0
    
    inorder_result = tr.inorder_traversal(root)
    print(f"Inorder: {inorder_result}")
    print(f"Max recursion depth: {tr.max_depth}")
    print()
    
    preorder_result = tr.preorder_traversal(root)
    print(f"Preorder: {preorder_result}")
    print()
    
    postorder_result = tr.postorder_traversal(root)
    print(f"Postorder: {postorder_result}")
    print()
    
    level_order_result = tr.level_order_recursive(root)
    print(f"Level order: {level_order_result}")
    print()
    
    # 2. Tree Properties
    print("=== TREE PROPERTIES ===")
    height = tr.tree_height(root)
    print(f"Tree height: {height}")
    print()
    
    size = tr.tree_size(root)
    print(f"Tree size: {size}")
    print()
    
    tree_sum = tr.tree_sum(root)
    print(f"Tree sum: {tree_sum}")
    print()
    
    is_balanced, _ = tr.is_balanced(root)
    print(f"Is balanced: {is_balanced}")
    print()
    
    # 3. Tree Construction
    print("=== TREE CONSTRUCTION ===")
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    
    print(f"Building tree from preorder {preorder} and inorder {inorder}:")
    constructed_tree = tr.build_tree_from_preorder_inorder(preorder, inorder)
    print("Constructed tree structure:")
    tr.print_tree_structure(constructed_tree)
    print()
    
    # 4. Tree Search and Paths
    print("=== TREE SEARCH AND PATHS ===")
    target = 5
    path = tr.find_path_to_node(root, target)
    print(f"Path to node {target}: {path}")
    print()
    
    all_paths = tr.all_root_to_leaf_paths(root)
    print(f"All root-to-leaf paths: {all_paths}")
    print()
    
    target_sum = 7
    has_sum = tr.path_sum(root, target_sum)
    print(f"Path with sum {target_sum} exists: {has_sum}")
    print()
    
    # 5. Lowest Common Ancestor
    print("=== LOWEST COMMON ANCESTOR ===")
    node_p = root.left.left  # Node 4
    node_q = root.left.right  # Node 5
    lca = tr.lowest_common_ancestor(root, node_p, node_q)
    print(f"LCA of {node_p.val} and {node_q.val}: {lca.val if lca else None}")
    print()
    
    # 6. Tree Modifications
    print("=== TREE MODIFICATIONS ===")
    print("Original tree:")
    tr.print_tree_structure(root)
    
    # Create a copy for inversion
    inverted_root = tr.build_tree_from_array([1, 2, 3, 4, 5])
    print("\nInverting tree...")
    tr.invert_tree(inverted_root)
    print("Inverted tree:")
    tr.print_tree_structure(inverted_root)

if __name__ == "__main__":
    demonstrate_tree_recursion()
    
    print("\n=== TREE RECURSION PATTERNS ===")
    print("🌳 Tree Traversal Pattern:")
    print("   - Process current node")
    print("   - Recurse on left subtree")
    print("   - Recurse on right subtree")
    
    print("\n🌳 Tree Property Pattern:")
    print("   - Base case: null node")
    print("   - Get property from left and right")
    print("   - Combine with current node")
    
    print("\n🌳 Tree Search Pattern:")
    print("   - Check current node")
    print("   - Search left if needed")
    print("   - Search right if needed")
    print("   - Backtrack if necessary")
    
    print("\n🌳 Tree Construction Pattern:")
    print("   - Create current node")
    print("   - Recursively build left subtree")
    print("   - Recursively build right subtree")
    
    print("\n=== TREE RECURSION TIPS ===")
    print("✅ Always handle null/empty cases first")
    print("✅ Think about what each recursive call returns")
    print("✅ Consider both top-down and bottom-up approaches")
    print("✅ Use helper functions for additional parameters")
    print("✅ Draw out small examples to understand the pattern")
    print("✅ Remember that tree height affects space complexity")
