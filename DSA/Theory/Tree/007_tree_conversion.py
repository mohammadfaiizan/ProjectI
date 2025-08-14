"""
Tree Conversion Algorithms - Converting Trees to Other Data Structures
This module implements various tree conversion algorithms and serialization techniques.
"""

from collections import deque
from typing import List, Optional, Union
import json

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class ListNode:
    """Linked list node structure"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class DLLNode:
    """Doubly linked list node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left  # Previous node
        self.right = right  # Next node
    
    def __repr__(self):
        return f"DLLNode({self.val})"

class TreeConversion:
    
    def __init__(self):
        """Initialize tree conversion algorithms"""
        pass
    
    # ==================== BINARY TREE TO DOUBLY LINKED LIST ====================
    
    def tree_to_dll_inorder(self, root: Optional[TreeNode]) -> Optional[DLLNode]:
        """
        Convert binary tree to doubly linked list using inorder traversal
        
        Time Complexity: O(n)
        Space Complexity: O(h) for recursion stack
        
        Args:
            root: Root of binary tree
        
        Returns:
            DLLNode: Head of doubly linked list
        """
        if not root:
            return None
        
        self.dll_head = None
        self.dll_prev = None
        
        def inorder_to_dll(node):
            if not node:
                return
            
            # Process left subtree
            inorder_to_dll(node.left)
            
            # Convert current node
            dll_node = DLLNode(node.val)
            
            if not self.dll_head:
                self.dll_head = dll_node
            else:
                self.dll_prev.right = dll_node
                dll_node.left = self.dll_prev
            
            self.dll_prev = dll_node
            
            # Process right subtree
            inorder_to_dll(node.right)
        
        inorder_to_dll(root)
        return self.dll_head
    
    def tree_to_dll_divide_conquer(self, root: Optional[TreeNode]) -> Optional[DLLNode]:
        """
        Convert binary tree to DLL using divide and conquer approach
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            DLLNode: Head of doubly linked list
        """
        def convert_to_dll(node):
            if not node:
                return None, None  # head, tail
            
            # Convert node
            dll_node = DLLNode(node.val)
            
            # Base case: leaf node
            if not node.left and not node.right:
                return dll_node, dll_node
            
            head, tail = dll_node, dll_node
            
            # Process left subtree
            if node.left:
                left_head, left_tail = convert_to_dll(node.left)
                left_tail.right = dll_node
                dll_node.left = left_tail
                head = left_head
            
            # Process right subtree
            if node.right:
                right_head, right_tail = convert_to_dll(node.right)
                dll_node.right = right_head
                right_head.left = dll_node
                tail = right_tail
            
            return head, tail
        
        if not root:
            return None
        
        head, _ = convert_to_dll(root)
        return head
    
    def tree_to_dll_morris_like(self, root: Optional[TreeNode]) -> Optional[DLLNode]:
        """
        Convert binary tree to DLL with O(1) space (Morris-like approach)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            root: Root of binary tree
        
        Returns:
            DLLNode: Head of doubly linked list
        """
        if not root:
            return None
        
        # Convert tree nodes to DLL nodes first
        def tree_to_dll_nodes(node):
            if not node:
                return None
            
            dll_node = DLLNode(node.val)
            if node.left:
                dll_node.left = tree_to_dll_nodes(node.left)
            if node.right:
                dll_node.right = tree_to_dll_nodes(node.right)
            
            return dll_node
        
        dll_root = tree_to_dll_nodes(root)
        
        # Now convert to proper DLL using Morris-like traversal
        current = dll_root
        head = None
        prev = None
        
        while current:
            if not current.left:
                # Process current node
                if not head:
                    head = current
                if prev:
                    prev.right = current
                    current.left = prev
                prev = current
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Make threading
                    predecessor.right = current
                    current = current.left
                else:
                    # Remove threading and process current
                    predecessor.right = None
                    if not head:
                        head = current
                    if prev:
                        prev.right = current
                        current.left = prev
                    prev = current
                    current = current.right
        
        return head
    
    # ==================== FLATTEN BINARY TREE TO LINKED LIST ====================
    
    def flatten_tree_preorder_recursive(self, root: Optional[TreeNode]) -> None:
        """
        Flatten binary tree to linked list using preorder traversal (LeetCode 114)
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree (modified in-place)
        """
        def flatten_helper(node):
            if not node:
                return None
            
            # Flatten left and right subtrees
            left_tail = flatten_helper(node.left)
            right_tail = flatten_helper(node.right)
            
            # If there's a left subtree, rearrange connections
            if node.left:
                left_tail.right = node.right
                node.right = node.left
                node.left = None
            
            # Return the tail of the flattened tree
            return right_tail or left_tail or node
        
        flatten_helper(root)
    
    def flatten_tree_iterative(self, root: Optional[TreeNode]) -> None:
        """
        Flatten binary tree iteratively using stack
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree (modified in-place)
        """
        if not root:
            return
        
        stack = [root]
        
        while stack:
            current = stack.pop()
            
            # Push right first, then left (reverse preorder)
            if current.right:
                stack.append(current.right)
            if current.left:
                stack.append(current.left)
            
            # Connect to next node
            if stack:
                current.right = stack[-1]
            
            current.left = None
    
    def flatten_tree_morris(self, root: Optional[TreeNode]) -> None:
        """
        Flatten binary tree using Morris-like approach (O(1) space)
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Args:
            root: Root of binary tree (modified in-place)
        """
        current = root
        
        while current:
            if current.left:
                # Find the rightmost node in left subtree
                predecessor = current.left
                while predecessor.right:
                    predecessor = predecessor.right
                
                # Connect predecessor to current's right subtree
                predecessor.right = current.right
                
                # Move left subtree to right
                current.right = current.left
                current.left = None
            
            current = current.right
    
    def flatten_to_linked_list(self, root: Optional[TreeNode]) -> Optional[ListNode]:
        """
        Flatten binary tree to actual linked list (not in-place)
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            ListNode: Head of linked list
        """
        if not root:
            return None
        
        result = []
        
        def preorder(node):
            if not node:
                return
            result.append(node.val)
            preorder(node.left)
            preorder(node.right)
        
        preorder(root)
        
        # Create linked list
        if not result:
            return None
        
        head = ListNode(result[0])
        current = head
        
        for i in range(1, len(result)):
            current.next = ListNode(result[i])
            current = current.next
        
        return head
    
    # ==================== SERIALIZE AND DESERIALIZE BINARY TREE ====================
    
    def serialize_preorder(self, root: Optional[TreeNode]) -> str:
        """
        Serialize binary tree using preorder traversal
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: Serialized string
        """
        def preorder_serialize(node):
            if not node:
                vals.append("null")
                return
            
            vals.append(str(node.val))
            preorder_serialize(node.left)
            preorder_serialize(node.right)
        
        vals = []
        preorder_serialize(root)
        return ",".join(vals)
    
    def deserialize_preorder(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize binary tree from preorder string
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            data: Serialized string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        def preorder_deserialize():
            val = next(vals)
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.left = preorder_deserialize()
            node.right = preorder_deserialize()
            return node
        
        vals = iter(data.split(","))
        return preorder_deserialize()
    
    def serialize_level_order(self, root: Optional[TreeNode]) -> str:
        """
        Serialize binary tree using level order traversal
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: Serialized string
        """
        if not root:
            return ""
        
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
        
        return ",".join(result)
    
    def deserialize_level_order(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize binary tree from level order string
        
        Args:
            data: Serialized string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        if not data:
            return None
        
        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        i = 1
        
        while queue and i < len(vals):
            node = queue.popleft()
            
            # Left child
            if i < len(vals) and vals[i] != "null":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1
            
            # Right child
            if i < len(vals) and vals[i] != "null":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1
        
        return root
    
    def serialize_postorder(self, root: Optional[TreeNode]) -> str:
        """
        Serialize binary tree using postorder traversal
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: Serialized string
        """
        def postorder_serialize(node):
            if not node:
                vals.append("null")
                return
            
            postorder_serialize(node.left)
            postorder_serialize(node.right)
            vals.append(str(node.val))
        
        vals = []
        postorder_serialize(root)
        return ",".join(vals)
    
    def deserialize_postorder(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize binary tree from postorder string
        
        Args:
            data: Serialized string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        def postorder_deserialize():
            val = vals.pop()
            if val == "null":
                return None
            
            node = TreeNode(int(val))
            node.right = postorder_deserialize()
            node.left = postorder_deserialize()
            return node
        
        vals = data.split(",")
        return postorder_deserialize()
    
    # ==================== SERIALIZE AND DESERIALIZE BST ====================
    
    def serialize_bst_compact(self, root: Optional[TreeNode]) -> str:
        """
        Serialize BST in compact format (preorder only)
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            root: Root of BST
        
        Returns:
            str: Serialized string
        """
        def preorder(node):
            if not node:
                return
            vals.append(str(node.val))
            preorder(node.left)
            preorder(node.right)
        
        vals = []
        preorder(root)
        return ",".join(vals)
    
    def deserialize_bst_compact(self, data: str) -> Optional[TreeNode]:
        """
        Deserialize BST from compact format
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            data: Serialized string
        
        Returns:
            TreeNode: Root of deserialized BST
        """
        if not data:
            return None
        
        vals = list(map(int, data.split(",")))
        
        def build_bst(min_val, max_val):
            if not vals or vals[0] < min_val or vals[0] > max_val:
                return None
            
            val = vals.pop(0)
            node = TreeNode(val)
            node.left = build_bst(min_val, val)
            node.right = build_bst(val, max_val)
            return node
        
        return build_bst(float('-inf'), float('inf'))
    
    def serialize_bst_bounds(self, root: Optional[TreeNode]) -> str:
        """
        Serialize BST with bounds information
        
        Args:
            root: Root of BST
        
        Returns:
            str: Serialized string with bounds
        """
        def serialize_with_bounds(node, min_val, max_val):
            if not node:
                return []
            
            result = [str(node.val)]
            result.extend(serialize_with_bounds(node.left, min_val, node.val))
            result.extend(serialize_with_bounds(node.right, node.val, max_val))
            return result
        
        vals = serialize_with_bounds(root, float('-inf'), float('inf'))
        return ",".join(vals)
    
    # ==================== JSON SERIALIZATION ====================
    
    def serialize_to_json(self, root: Optional[TreeNode]) -> str:
        """
        Serialize binary tree to JSON format
        
        Args:
            root: Root of binary tree
        
        Returns:
            str: JSON string
        """
        def tree_to_dict(node):
            if not node:
                return None
            
            return {
                'val': node.val,
                'left': tree_to_dict(node.left),
                'right': tree_to_dict(node.right)
            }
        
        tree_dict = tree_to_dict(root)
        return json.dumps(tree_dict, indent=2)
    
    def deserialize_from_json(self, json_str: str) -> Optional[TreeNode]:
        """
        Deserialize binary tree from JSON format
        
        Args:
            json_str: JSON string
        
        Returns:
            TreeNode: Root of deserialized tree
        """
        def dict_to_tree(tree_dict):
            if not tree_dict:
                return None
            
            node = TreeNode(tree_dict['val'])
            node.left = dict_to_tree(tree_dict['left'])
            node.right = dict_to_tree(tree_dict['right'])
            return node
        
        tree_dict = json.loads(json_str)
        return dict_to_tree(tree_dict)
    
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
    
    def print_dll(self, head: Optional[DLLNode], max_nodes: int = 20):
        """Print doubly linked list"""
        if not head:
            print("Empty DLL")
            return
        
        print("DLL (forward):", end=" ")
        current = head
        count = 0
        
        while current and count < max_nodes:
            print(current.val, end="")
            if current.right:
                print(" <-> ", end="")
            current = current.right
            count += 1
        
        if current:
            print(" ... (truncated)")
        else:
            print()
    
    def print_linked_list(self, head: Optional[ListNode], max_nodes: int = 20):
        """Print linked list"""
        if not head:
            print("Empty list")
            return
        
        print("Linked list:", end=" ")
        current = head
        count = 0
        
        while current and count < max_nodes:
            print(current.val, end="")
            if current.next:
                print(" -> ", end="")
            current = current.next
            count += 1
        
        if current:
            print(" ... (truncated)")
        else:
            print()
    
    def verify_dll(self, head: Optional[DLLNode]) -> bool:
        """Verify doubly linked list integrity"""
        if not head:
            return True
        
        # Check forward links
        current = head
        while current.right:
            if current.right.left != current:
                return False
            current = current.right
        
        # Check backward links
        while current.left:
            if current.left.right != current:
                return False
            current = current.left
        
        return current == head


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Tree Conversion Demo ===\n")
    
    converter = TreeConversion()
    
    # Build test tree
    arr = [1, 2, 5, 3, 4, None, 6]
    root = converter.build_tree_from_array(arr)
    
    print("Original Tree Structure:")
    converter.print_tree_structure(root)
    print()
    
    # Example 1: Binary Tree to Doubly Linked List
    print("1. Binary Tree to Doubly Linked List:")
    
    # Method 1: Inorder traversal
    dll_head_inorder = converter.tree_to_dll_inorder(root)
    print("DLL (inorder traversal):")
    converter.print_dll(dll_head_inorder)
    print(f"DLL integrity check: {converter.verify_dll(dll_head_inorder)}")
    
    # Method 2: Divide and conquer
    dll_head_dc = converter.tree_to_dll_divide_conquer(root)
    print("DLL (divide & conquer):")
    converter.print_dll(dll_head_dc)
    print()
    
    # Example 2: Flatten Binary Tree to Linked List
    print("2. Flatten Binary Tree to Linked List:")
    
    # Test with a copy of the tree for each method
    test_trees = [
        converter.build_tree_from_array(arr),
        converter.build_tree_from_array(arr),
        converter.build_tree_from_array(arr)
    ]
    
    print("Before flattening:")
    converter.print_tree_structure(test_trees[0])
    
    # Method 1: Recursive
    converter.flatten_tree_preorder_recursive(test_trees[0])
    print("After recursive flattening:")
    converter.print_tree_structure(test_trees[0])
    
    # Method 2: Iterative
    converter.flatten_tree_iterative(test_trees[1])
    print("After iterative flattening:")
    converter.print_tree_structure(test_trees[1])
    
    # Method 3: Morris approach
    converter.flatten_tree_morris(test_trees[2])
    print("After Morris flattening:")
    converter.print_tree_structure(test_trees[2])
    
    # Method 4: To actual linked list
    linked_list = converter.flatten_to_linked_list(root)
    converter.print_linked_list(linked_list)
    print()
    
    # Example 3: Serialize and Deserialize Binary Tree
    print("3. Serialize and Deserialize Binary Tree:")
    
    # Preorder serialization
    serialized_preorder = converter.serialize_preorder(root)
    print(f"Preorder serialized: {serialized_preorder}")
    
    deserialized_preorder = converter.deserialize_preorder(serialized_preorder)
    print("Deserialized tree (preorder):")
    converter.print_tree_structure(deserialized_preorder)
    
    # Level order serialization
    serialized_level = converter.serialize_level_order(root)
    print(f"Level order serialized: {serialized_level}")
    
    deserialized_level = converter.deserialize_level_order(serialized_level)
    print("Deserialized tree (level order):")
    converter.print_tree_structure(deserialized_level)
    
    # Postorder serialization
    serialized_postorder = converter.serialize_postorder(root)
    print(f"Postorder serialized: {serialized_postorder}")
    
    deserialized_postorder = converter.deserialize_postorder(serialized_postorder)
    print("Deserialized tree (postorder):")
    converter.print_tree_structure(deserialized_postorder)
    print()
    
    # Example 4: BST Serialization
    print("4. BST Serialization:")
    
    # Build a BST
    bst_arr = [8, 3, 10, 1, 6, None, 14, None, None, 4, 7, 13]
    bst_root = converter.build_tree_from_array(bst_arr)
    
    print("BST Structure:")
    converter.print_tree_structure(bst_root)
    
    # Compact BST serialization
    bst_serialized = converter.serialize_bst_compact(bst_root)
    print(f"BST compact serialized: {bst_serialized}")
    
    bst_deserialized = converter.deserialize_bst_compact(bst_serialized)
    print("Deserialized BST:")
    converter.print_tree_structure(bst_deserialized)
    print()
    
    # Example 5: JSON Serialization
    print("5. JSON Serialization:")
    
    json_serialized = converter.serialize_to_json(root)
    print("JSON serialized:")
    print(json_serialized)
    
    json_deserialized = converter.deserialize_from_json(json_serialized)
    print("Deserialized from JSON:")
    converter.print_tree_structure(json_deserialized)
    print()
    
    # Example 6: Complex Tree Conversion
    print("6. Complex Tree Conversion Example:")
    
    # Build a more complex tree
    complex_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    complex_root = converter.build_tree_from_array(complex_arr)
    
    print("Complex Tree Structure:")
    converter.print_tree_structure(complex_root)
    
    # Convert to DLL
    complex_dll = converter.tree_to_dll_inorder(complex_root)
    print("Complex tree as DLL:")
    converter.print_dll(complex_dll)
    
    # Serialize using different methods
    methods = [
        ("Preorder", converter.serialize_preorder),
        ("Level Order", converter.serialize_level_order),
        ("Postorder", converter.serialize_postorder)
    ]
    
    print("Serialization comparison:")
    for name, method in methods:
        serialized = method(complex_root)
        print(f"{name}: {serialized}")
    
    print("\n=== Demo Complete ===") 