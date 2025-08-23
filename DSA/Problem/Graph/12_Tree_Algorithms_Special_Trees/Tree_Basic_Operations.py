"""
Tree Basic Operations - Comprehensive Implementation
Difficulty: Easy

This file provides fundamental tree operations and algorithms that form the 
foundation for more advanced tree algorithms. Covers basic traversals, 
construction, validation, and utility operations.

Key Concepts:
1. Tree Construction and Representation
2. Basic Tree Traversals (Inorder, Preorder, Postorder)
3. Level Order Traversal
4. Tree Validation and Properties
5. Tree Utility Operations
6. Tree Serialization and Deserialization
"""

from typing import Optional, List, Dict, Set, Tuple, Union
from collections import deque, defaultdict
import json

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeBasicOperations:
    """Comprehensive implementation of basic tree operations"""
    
    def __init__(self):
        self.traversal_result = []
    
    # ==================== TREE TRAVERSALS ====================
    
    def inorder_traversal_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 1: Recursive Inorder Traversal (Left -> Root -> Right)
        
        Time: O(n), Space: O(h)
        """
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    def inorder_traversal_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 2: Iterative Inorder Traversal using Stack
        
        Time: O(n), Space: O(h)
        """
        result = []
        stack = []
        current = root
        
        while stack or current:
            # Go to leftmost node
            while current:
                stack.append(current)
                current = current.left
            
            # Process current node
            current = stack.pop()
            result.append(current.val)
            
            # Move to right subtree
            current = current.right
        
        return result
    
    def preorder_traversal_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 3: Recursive Preorder Traversal (Root -> Left -> Right)
        
        Time: O(n), Space: O(h)
        """
        result = []
        
        def preorder(node):
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        return result
    
    def preorder_traversal_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 4: Iterative Preorder Traversal using Stack
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return []
        
        result = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            result.append(node.val)
            
            # Push right first, then left (stack is LIFO)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        
        return result
    
    def postorder_traversal_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 5: Recursive Postorder Traversal (Left -> Right -> Root)
        
        Time: O(n), Space: O(h)
        """
        result = []
        
        def postorder(node):
            if node:
                postorder(node.left)
                postorder(node.right)
                result.append(node.val)
        
        postorder(root)
        return result
    
    def postorder_traversal_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 6: Iterative Postorder Traversal using Stack
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return []
        
        result = []
        stack = []
        current = root
        last_visited = None
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                
                # If right child exists and hasn't been processed yet
                if peek_node.right and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    result.append(peek_node.val)
                    last_visited = stack.pop()
        
        return result
    
    def level_order_traversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Approach 7: Level Order Traversal (BFS)
        
        Time: O(n), Space: O(w) where w is maximum width
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
        
        return result
    
    def level_order_traversal_flat(self, root: Optional[TreeNode]) -> List[int]:
        """
        Approach 8: Level Order Traversal (Flat Result)
        
        Time: O(n), Space: O(w)
        """
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
    
    def build_tree_from_preorder_inorder(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        Approach 9: Build Tree from Preorder and Inorder Traversals
        
        Time: O(n), Space: O(n)
        """
        if not preorder or not inorder:
            return None
        
        # Create index map for inorder for O(1) lookup
        inorder_map = {val: i for i, val in enumerate(inorder)}
        self.preorder_idx = 0
        
        def build(left, right):
            if left > right:
                return None
            
            # Root is first element in preorder
            root_val = preorder[self.preorder_idx]
            self.preorder_idx += 1
            
            root = TreeNode(root_val)
            
            # Find root position in inorder
            root_idx = inorder_map[root_val]
            
            # Build left subtree first (preorder: root -> left -> right)
            root.left = build(left, root_idx - 1)
            root.right = build(root_idx + 1, right)
            
            return root
        
        return build(0, len(inorder) - 1)
    
    def build_tree_from_postorder_inorder(self, postorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        Approach 10: Build Tree from Postorder and Inorder Traversals
        
        Time: O(n), Space: O(n)
        """
        if not postorder or not inorder:
            return None
        
        inorder_map = {val: i for i, val in enumerate(inorder)}
        self.postorder_idx = len(postorder) - 1
        
        def build(left, right):
            if left > right:
                return None
            
            # Root is last element in postorder
            root_val = postorder[self.postorder_idx]
            self.postorder_idx -= 1
            
            root = TreeNode(root_val)
            root_idx = inorder_map[root_val]
            
            # Build right subtree first (postorder: left -> right -> root)
            root.right = build(root_idx + 1, right)
            root.left = build(left, root_idx - 1)
            
            return root
        
        return build(0, len(inorder) - 1)
    
    def build_tree_from_level_order(self, level_order: List[Optional[int]]) -> Optional[TreeNode]:
        """
        Approach 11: Build Tree from Level Order Traversal
        
        Time: O(n), Space: O(n)
        """
        if not level_order or level_order[0] is None:
            return None
        
        root = TreeNode(level_order[0])
        queue = deque([root])
        i = 1
        
        while queue and i < len(level_order):
            node = queue.popleft()
            
            # Add left child
            if i < len(level_order) and level_order[i] is not None:
                node.left = TreeNode(level_order[i])
                queue.append(node.left)
            i += 1
            
            # Add right child
            if i < len(level_order) and level_order[i] is not None:
                node.right = TreeNode(level_order[i])
                queue.append(node.right)
            i += 1
        
        return root
    
    # ==================== TREE VALIDATION ====================
    
    def is_valid_bst(self, root: Optional[TreeNode]) -> bool:
        """
        Approach 12: Validate Binary Search Tree
        
        Time: O(n), Space: O(h)
        """
        def validate(node, min_val, max_val):
            if not node:
                return True
            
            if node.val <= min_val or node.val >= max_val:
                return False
            
            return (validate(node.left, min_val, node.val) and
                    validate(node.right, node.val, max_val))
        
        return validate(root, float('-inf'), float('inf'))
    
    def is_balanced(self, root: Optional[TreeNode]) -> bool:
        """
        Approach 13: Check if Tree is Height-Balanced
        
        Time: O(n), Space: O(h)
        """
        def check_balance(node):
            if not node:
                return 0, True
            
            left_height, left_balanced = check_balance(node.left)
            if not left_balanced:
                return 0, False
            
            right_height, right_balanced = check_balance(node.right)
            if not right_balanced:
                return 0, False
            
            height_diff = abs(left_height - right_height)
            is_balanced = height_diff <= 1
            
            return max(left_height, right_height) + 1, is_balanced
        
        _, balanced = check_balance(root)
        return balanced
    
    def is_symmetric(self, root: Optional[TreeNode]) -> bool:
        """
        Approach 14: Check if Tree is Symmetric
        
        Time: O(n), Space: O(h)
        """
        def is_mirror(left, right):
            if not left and not right:
                return True
            if not left or not right:
                return False
            
            return (left.val == right.val and
                    is_mirror(left.left, right.right) and
                    is_mirror(left.right, right.left))
        
        if not root:
            return True
        
        return is_mirror(root.left, root.right)
    
    # ==================== TREE PROPERTIES ====================
    
    def get_tree_height(self, root: Optional[TreeNode]) -> int:
        """
        Approach 15: Calculate Tree Height
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        return 1 + max(self.get_tree_height(root.left),
                       self.get_tree_height(root.right))
    
    def count_nodes(self, root: Optional[TreeNode]) -> int:
        """
        Approach 16: Count Total Nodes
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        return 1 + self.count_nodes(root.left) + self.count_nodes(root.right)
    
    def count_leaves(self, root: Optional[TreeNode]) -> int:
        """
        Approach 17: Count Leaf Nodes
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        if not root.left and not root.right:
            return 1
        
        return self.count_leaves(root.left) + self.count_leaves(root.right)
    
    def get_tree_width(self, root: Optional[TreeNode]) -> int:
        """
        Approach 18: Get Maximum Width of Tree
        
        Time: O(n), Space: O(w)
        """
        if not root:
            return 0
        
        max_width = 0
        queue = deque([root])
        
        while queue:
            level_width = len(queue)
            max_width = max(max_width, level_width)
            
            for _ in range(level_width):
                node = queue.popleft()
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return max_width
    
    # ==================== TREE SERIALIZATION ====================
    
    def serialize(self, root: Optional[TreeNode]) -> str:
        """
        Approach 19: Serialize Tree to String
        
        Time: O(n), Space: O(n)
        """
        def preorder_serialize(node):
            if not node:
                return "null"
            
            return f"{node.val},{preorder_serialize(node.left)},{preorder_serialize(node.right)}"
        
        return preorder_serialize(root)
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """
        Approach 20: Deserialize String to Tree
        
        Time: O(n), Space: O(n)
        """
        def preorder_deserialize():
            val = next(values)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = preorder_deserialize()
            node.right = preorder_deserialize()
            
            return node
        
        values = iter(data.split(','))
        return preorder_deserialize()
    
    def serialize_level_order(self, root: Optional[TreeNode]) -> str:
        """
        Approach 21: Serialize using Level Order
        
        Time: O(n), Space: O(n)
        """
        if not root:
            return "[]"
        
        result = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("null")
        
        # Remove trailing nulls
        while result and result[-1] == "null":
            result.pop()
        
        return "[" + ",".join(result) + "]"
    
    # ==================== UTILITY OPERATIONS ====================
    
    def find_path_to_node(self, root: Optional[TreeNode], target: int) -> List[int]:
        """
        Approach 22: Find Path from Root to Target Node
        
        Time: O(n), Space: O(h)
        """
        def find_path(node, path):
            if not node:
                return False
            
            path.append(node.val)
            
            if node.val == target:
                return True
            
            if (find_path(node.left, path) or 
                find_path(node.right, path)):
                return True
            
            path.pop()
            return False
        
        path = []
        find_path(root, path)
        return path
    
    def lowest_common_ancestor(self, root: Optional[TreeNode], p: int, q: int) -> Optional[TreeNode]:
        """
        Approach 23: Find Lowest Common Ancestor
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return None
        
        if root.val == p or root.val == q:
            return root
        
        left_lca = self.lowest_common_ancestor(root.left, p, q)
        right_lca = self.lowest_common_ancestor(root.right, p, q)
        
        if left_lca and right_lca:
            return root
        
        return left_lca if left_lca else right_lca
    
    def tree_to_list_representation(self, root: Optional[TreeNode]) -> List[Optional[int]]:
        """
        Approach 24: Convert Tree to List Representation
        
        Time: O(n), Space: O(n)
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

def create_sample_trees():
    """Create sample trees for testing"""
    # Tree 1: Balanced BST [4,2,6,1,3,5,7]
    tree1 = TreeNode(4)
    tree1.left = TreeNode(2)
    tree1.right = TreeNode(6)
    tree1.left.left = TreeNode(1)
    tree1.left.right = TreeNode(3)
    tree1.right.left = TreeNode(5)
    tree1.right.right = TreeNode(7)
    
    # Tree 2: Unbalanced tree [1,2,3,4,5]
    tree2 = TreeNode(1)
    tree2.right = TreeNode(2)
    tree2.right.left = TreeNode(3)
    tree2.right.left.right = TreeNode(4)
    tree2.right.left.right.left = TreeNode(5)
    
    # Tree 3: Single node
    tree3 = TreeNode(42)
    
    return [
        (tree1, "Balanced BST"),
        (tree2, "Unbalanced tree"),
        (tree3, "Single node"),
        (None, "Empty tree")
    ]

def test_tree_operations():
    """Test all tree operations"""
    ops = TreeBasicOperations()
    sample_trees = create_sample_trees()
    
    print("=== Testing Tree Basic Operations ===")
    
    for tree, description in sample_trees:
        print(f"\n--- {description} ---")
        
        if tree:
            # Test traversals
            print(f"Inorder (Recursive):  {ops.inorder_traversal_recursive(tree)}")
            print(f"Inorder (Iterative):  {ops.inorder_traversal_iterative(tree)}")
            print(f"Preorder (Recursive): {ops.preorder_traversal_recursive(tree)}")
            print(f"Preorder (Iterative): {ops.preorder_traversal_iterative(tree)}")
            print(f"Postorder (Recursive):{ops.postorder_traversal_recursive(tree)}")
            print(f"Level Order:          {ops.level_order_traversal_flat(tree)}")
            
            # Test properties
            print(f"Height: {ops.get_tree_height(tree)}")
            print(f"Node count: {ops.count_nodes(tree)}")
            print(f"Leaf count: {ops.count_leaves(tree)}")
            print(f"Max width: {ops.get_tree_width(tree)}")
            
            # Test validations
            print(f"Is BST: {ops.is_valid_bst(tree)}")
            print(f"Is balanced: {ops.is_balanced(tree)}")
            print(f"Is symmetric: {ops.is_symmetric(tree)}")
            
            # Test serialization
            serialized = ops.serialize(tree)
            print(f"Serialized: {serialized[:50]}...")
            
        else:
            print("Empty tree - skipping detailed tests")

def demonstrate_tree_construction():
    """Demonstrate tree construction from traversals"""
    print("\n=== Tree Construction Demo ===")
    
    ops = TreeBasicOperations()
    
    # Example traversals
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    postorder = [9, 15, 7, 20, 3]
    
    print(f"Given traversals:")
    print(f"Preorder:  {preorder}")
    print(f"Inorder:   {inorder}")
    print(f"Postorder: {postorder}")
    
    # Build from preorder + inorder
    tree1 = ops.build_tree_from_preorder_inorder(preorder, inorder)
    print(f"\nBuilt from preorder + inorder:")
    print(f"Level order: {ops.level_order_traversal_flat(tree1)}")
    
    # Build from postorder + inorder
    tree2 = ops.build_tree_from_postorder_inorder(postorder, inorder)
    print(f"\nBuilt from postorder + inorder:")
    print(f"Level order: {ops.level_order_traversal_flat(tree2)}")
    
    # Build from level order
    level_order = [3, 9, 20, None, None, 15, 7]
    tree3 = ops.build_tree_from_level_order(level_order)
    print(f"\nBuilt from level order {level_order}:")
    print(f"Inorder: {ops.inorder_traversal_recursive(tree3)}")

def analyze_tree_operations_complexity():
    """Analyze complexity of tree operations"""
    print("\n=== Complexity Analysis ===")
    
    print("Traversal Operations:")
    print("• Inorder/Preorder/Postorder: O(n) time, O(h) space")
    print("• Level Order: O(n) time, O(w) space")
    print("• All traversals visit each node exactly once")
    
    print("\nTree Construction:")
    print("• From traversals: O(n) time, O(n) space")
    print("• Requires inorder + one other traversal")
    print("• Level order construction: O(n) time, O(n) space")
    
    print("\nValidation Operations:")
    print("• BST validation: O(n) time, O(h) space")
    print("• Balance check: O(n) time, O(h) space")
    print("• Symmetry check: O(n) time, O(h) space")
    
    print("\nProperty Calculations:")
    print("• Height: O(n) time, O(h) space")
    print("• Node/Leaf count: O(n) time, O(h) space")
    print("• Width: O(n) time, O(w) space")
    
    print("\nUtility Operations:")
    print("• Serialization: O(n) time, O(n) space")
    print("• Path finding: O(n) time, O(h) space")
    print("• LCA: O(n) time, O(h) space")

if __name__ == "__main__":
    test_tree_operations()
    demonstrate_tree_construction()
    analyze_tree_operations_complexity()

"""
Tree Basic Operations - Key Insights:

1. **Fundamental Operations:**
   - Tree traversals form the basis for most tree algorithms
   - Construction from traversals requires understanding of order
   - Validation ensures tree properties are maintained
   - Utility operations support complex algorithms

2. **Traversal Strategies:**
   - Recursive: Simple but limited by stack depth
   - Iterative: More control, handles deep trees
   - Level order: Natural for width-based problems
   - Each traversal has specific use cases

3. **Tree Construction:**
   - Inorder + one other traversal uniquely determines tree
   - Level order construction is intuitive and efficient
   - Proper handling of null nodes is crucial
   - Index mapping optimizes construction time

4. **Validation Techniques:**
   - BST: Maintain min/max bounds during traversal
   - Balance: Check height differences recursively
   - Symmetry: Compare left and right subtrees
   - Early termination improves efficiency

5. **Practical Applications:**
   - Foundation for advanced tree algorithms
   - Database indexing and search operations
   - Expression parsing and evaluation
   - File system and hierarchical data representation

These basic operations provide the building blocks
for all advanced tree algorithms and data structures.
"""
