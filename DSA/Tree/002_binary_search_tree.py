"""
Binary Search Tree (BST) - Complete BST Operations and Algorithms
This module implements all BST operations, validation, and conversion algorithms.
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

class ListNode:
    """Linked list node for BST to list conversion"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"ListNode({self.val})"

class BinarySearchTree:
    
    def __init__(self, root=None):
        """Initialize BST with optional root"""
        self.root = root
    
    # ==================== BASIC BST OPERATIONS ====================
    
    def insert(self, root: Optional[TreeNode], val: int) -> TreeNode:
        """
        Insert a value into BST
        
        Time Complexity: O(h) where h is height, O(log n) average, O(n) worst
        Space Complexity: O(h) for recursion stack
        
        Args:
            root: Root of BST
            val: Value to insert
        
        Returns:
            TreeNode: Root of modified BST
        """
        if not root:
            return TreeNode(val)
        
        if val < root.val:
            root.left = self.insert(root.left, val)
        elif val > root.val:
            root.right = self.insert(root.right, val)
        # If val == root.val, we don't insert duplicates
        
        return root
    
    def insert_iterative(self, root: Optional[TreeNode], val: int) -> TreeNode:
        """Iterative insertion into BST"""
        if not root:
            return TreeNode(val)
        
        current = root
        
        while True:
            if val < current.val:
                if current.left:
                    current = current.left
                else:
                    current.left = TreeNode(val)
                    break
            elif val > current.val:
                if current.right:
                    current = current.right
                else:
                    current.right = TreeNode(val)
                    break
            else:
                # Duplicate value, don't insert
                break
        
        return root
    
    def search(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        Search for a value in BST
        
        Time Complexity: O(h)
        Space Complexity: O(h) for recursion
        
        Args:
            root: Root of BST
            val: Value to search
        
        Returns:
            TreeNode: Node with the value, None if not found
        """
        if not root or root.val == val:
            return root
        
        if val < root.val:
            return self.search(root.left, val)
        else:
            return self.search(root.right, val)
    
    def search_iterative(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """Iterative search in BST"""
        while root and root.val != val:
            if val < root.val:
                root = root.left
            else:
                root = root.right
        
        return root
    
    def delete_node(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        Delete a node from BST
        
        Time Complexity: O(h)
        Space Complexity: O(h)
        
        Args:
            root: Root of BST
            val: Value to delete
        
        Returns:
            TreeNode: Root of modified BST
        """
        if not root:
            return root
        
        if val < root.val:
            root.left = self.delete_node(root.left, val)
        elif val > root.val:
            root.right = self.delete_node(root.right, val)
        else:
            # Node to be deleted found
            
            # Case 1: Node with only right child or no child
            if not root.left:
                return root.right
            
            # Case 2: Node with only left child
            if not root.right:
                return root.left
            
            # Case 3: Node with two children
            # Find inorder successor (smallest in right subtree)
            successor = self.find_min(root.right)
            root.val = successor.val
            # Delete the inorder successor
            root.right = self.delete_node(root.right, successor.val)
        
        return root
    
    def find_min(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """Find minimum value node in BST"""
        if not root:
            return None
        
        while root.left:
            root = root.left
        
        return root
    
    def find_max(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """Find maximum value node in BST"""
        if not root:
            return None
        
        while root.right:
            root = root.right
        
        return root
    
    # ==================== BST VALIDATION ====================
    
    def is_valid_bst(self, root: Optional[TreeNode]) -> bool:
        """
        Validate if tree is a valid BST
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of tree to validate
        
        Returns:
            bool: True if valid BST
        """
        def validate(node, min_val, max_val):
            if not node:
                return True
            
            if node.val <= min_val or node.val >= max_val:
                return False
            
            return (validate(node.left, min_val, node.val) and
                   validate(node.right, node.val, max_val))
        
        return validate(root, float('-inf'), float('inf'))
    
    def is_valid_bst_inorder(self, root: Optional[TreeNode]) -> bool:
        """Validate BST using inorder traversal"""
        def inorder(node):
            if not node:
                return True
            
            if not inorder(node.left):
                return False
            
            if self.prev[0] is not None and node.val <= self.prev[0]:
                return False
            
            self.prev[0] = node.val
            
            return inorder(node.right)
        
        self.prev = [None]
        return inorder(root)
    
    # ==================== KTH SMALLEST/LARGEST ====================
    
    def kth_smallest(self, root: Optional[TreeNode], k: int) -> int:
        """
        Find kth smallest element in BST
        
        Time Complexity: O(h + k)
        Space Complexity: O(h)
        
        Args:
            root: Root of BST
            k: Position (1-indexed)
        
        Returns:
            int: kth smallest value
        """
        def inorder(node):
            if not node:
                return None
            
            left_result = inorder(node.left)
            if left_result is not None:
                return left_result
            
            self.count += 1
            if self.count == k:
                return node.val
            
            return inorder(node.right)
        
        self.count = 0
        return inorder(root)
    
    def kth_largest(self, root: Optional[TreeNode], k: int) -> int:
        """Find kth largest element in BST (reverse inorder)"""
        def reverse_inorder(node):
            if not node:
                return None
            
            right_result = reverse_inorder(node.right)
            if right_result is not None:
                return right_result
            
            self.count += 1
            if self.count == k:
                return node.val
            
            return reverse_inorder(node.left)
        
        self.count = 0
        return reverse_inorder(root)
    
    def kth_smallest_iterative(self, root: Optional[TreeNode], k: int) -> int:
        """Iterative approach for kth smallest"""
        stack = []
        count = 0
        
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            count += 1
            
            if count == k:
                return root.val
            
            root = root.right
        
        return -1  # k is invalid
    
    # ==================== LOWEST COMMON ANCESTOR IN BST ====================
    
    def lowest_common_ancestor_bst(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Find LCA in BST (leverages BST property)
        
        Time Complexity: O(h)
        Space Complexity: O(h) for recursion
        
        Args:
            root: Root of BST
            p, q: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        if not root:
            return None
        
        # If both p and q are smaller than root, LCA is in left subtree
        if p.val < root.val and q.val < root.val:
            return self.lowest_common_ancestor_bst(root.left, p, q)
        
        # If both p and q are greater than root, LCA is in right subtree
        if p.val > root.val and q.val > root.val:
            return self.lowest_common_ancestor_bst(root.right, p, q)
        
        # If one is smaller and one is greater, root is LCA
        return root
    
    def lowest_common_ancestor_bst_iterative(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """Iterative LCA in BST"""
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
        
        return None
    
    # ==================== FLOOR AND CEIL IN BST ====================
    
    def floor_in_bst(self, root: Optional[TreeNode], val: int) -> Optional[int]:
        """
        Find floor of value in BST (largest value <= val)
        
        Args:
            root: Root of BST
            val: Target value
        
        Returns:
            int: Floor value, None if not found
        """
        floor_val = None
        
        while root:
            if root.val == val:
                return root.val
            elif root.val < val:
                floor_val = root.val
                root = root.right
            else:
                root = root.left
        
        return floor_val
    
    def ceil_in_bst(self, root: Optional[TreeNode], val: int) -> Optional[int]:
        """
        Find ceil of value in BST (smallest value >= val)
        
        Args:
            root: Root of BST
            val: Target value
        
        Returns:
            int: Ceil value, None if not found
        """
        ceil_val = None
        
        while root:
            if root.val == val:
                return root.val
            elif root.val > val:
                ceil_val = root.val
                root = root.left
            else:
                root = root.right
        
        return ceil_val
    
    # ==================== CONVERSION ALGORITHMS ====================
    
    def sorted_array_to_bst(self, nums: List[int]) -> Optional[TreeNode]:
        """
        Convert sorted array to balanced BST
        
        Time Complexity: O(n)
        Space Complexity: O(log n) for recursion
        
        Args:
            nums: Sorted array
        
        Returns:
            TreeNode: Root of balanced BST
        """
        def build_bst(left, right):
            if left > right:
                return None
            
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            
            root.left = build_bst(left, mid - 1)
            root.right = build_bst(mid + 1, right)
            
            return root
        
        return build_bst(0, len(nums) - 1)
    
    def bst_to_sorted_list(self, root: Optional[TreeNode]) -> Optional[ListNode]:
        """
        Convert BST to sorted doubly linked list
        
        Args:
            root: Root of BST
        
        Returns:
            ListNode: Head of sorted linked list
        """
        if not root:
            return None
        
        # Use inorder traversal to get sorted order
        values = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                values.append(node.val)
                inorder(node.right)
        
        inorder(root)
        
        # Build linked list
        if not values:
            return None
        
        head = ListNode(values[0])
        current = head
        
        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next
        
        return head
    
    def bst_to_sorted_list_inplace(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Convert BST to sorted linked list in-place (using tree nodes)
        
        Args:
            root: Root of BST
        
        Returns:
            TreeNode: Head of linked list (left=None, right=next)
        """
        if not root:
            return None
        
        def inorder(node):
            if not node:
                return
            
            inorder(node.left)
            
            if self.prev:
                self.prev.right = node
                node.left = self.prev
            else:
                self.head = node
            
            self.prev = node
            
            inorder(node.right)
        
        self.prev = None
        self.head = None
        inorder(root)
        
        # Convert to singly linked list format
        current = self.head
        while current:
            current.left = None
            current = current.right
        
        return self.head
    
    def bst_to_sorted_array(self, root: Optional[TreeNode]) -> List[int]:
        """Convert BST to sorted array using inorder traversal"""
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    # ==================== ADVANCED BST OPERATIONS ====================
    
    def find_range(self, root: Optional[TreeNode], low: int, high: int) -> List[int]:
        """
        Find all values in BST within given range [low, high]
        
        Args:
            root: Root of BST
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
        
        Returns:
            list: Values in range
        """
        result = []
        
        def inorder(node):
            if not node:
                return
            
            if node.val > low:
                inorder(node.left)
            
            if low <= node.val <= high:
                result.append(node.val)
            
            if node.val < high:
                inorder(node.right)
        
        inorder(root)
        return result
    
    def trim_bst(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        """
        Trim BST to only contain values in range [low, high]
        
        Args:
            root: Root of BST
            low: Lower bound
            high: Upper bound
        
        Returns:
            TreeNode: Root of trimmed BST
        """
        if not root:
            return None
        
        if root.val < low:
            return self.trim_bst(root.right, low, high)
        
        if root.val > high:
            return self.trim_bst(root.left, low, high)
        
        root.left = self.trim_bst(root.left, low, high)
        root.right = self.trim_bst(root.right, low, high)
        
        return root
    
    def find_successor(self, root: Optional[TreeNode], target: TreeNode) -> Optional[TreeNode]:
        """Find inorder successor of a node in BST"""
        successor = None
        
        while root:
            if target.val < root.val:
                successor = root
                root = root.left
            else:
                root = root.right
        
        return successor
    
    def find_predecessor(self, root: Optional[TreeNode], target: TreeNode) -> Optional[TreeNode]:
        """Find inorder predecessor of a node in BST"""
        predecessor = None
        
        while root:
            if target.val > root.val:
                predecessor = root
                root = root.right
            else:
                root = root.left
        
        return predecessor
    
    # ==================== UTILITY METHODS ====================
    
    def build_bst_from_list(self, values: List[int]) -> Optional[TreeNode]:
        """Build BST by inserting values sequentially"""
        root = None
        for val in values:
            root = self.insert(root, val)
        return root
    
    def inorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """Get inorder traversal of BST (should be sorted)"""
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    def print_tree_structure(self, root: Optional[TreeNode], level: int = 0, prefix: str = "Root: "):
        """Print BST structure"""
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
    
    def get_bst_height(self, root: Optional[TreeNode]) -> int:
        """Get height of BST"""
        if not root:
            return -1
        return 1 + max(self.get_bst_height(root.left), self.get_bst_height(root.right))
    
    def count_nodes_in_range(self, root: Optional[TreeNode], low: int, high: int) -> int:
        """Count nodes with values in range [low, high]"""
        if not root:
            return 0
        
        count = 0
        if low <= root.val <= high:
            count = 1
        
        if root.val > low:
            count += self.count_nodes_in_range(root.left, low, high)
        
        if root.val < high:
            count += self.count_nodes_in_range(root.right, low, high)
        
        return count


# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Binary Search Tree Demo ===\n")
    
    bst = BinarySearchTree()
    
    # Example 1: Building BST and basic operations
    print("1. BST Construction and Basic Operations:")
    values = [15, 10, 20, 8, 12, 16, 25]
    print(f"Inserting values: {values}")
    
    root = None
    for val in values:
        root = bst.insert(root, val)
    
    print("BST structure:")
    bst.print_tree_structure(root)
    
    print(f"Inorder traversal (should be sorted): {bst.inorder_traversal(root)}")
    print(f"BST height: {bst.get_bst_height(root)}")
    print()
    
    # Example 2: Search operations
    print("2. Search Operations:")
    search_vals = [12, 7, 20]
    for val in search_vals:
        found = bst.search(root, val)
        print(f"Search {val}: {'Found' if found else 'Not found'}")
    
    print(f"Minimum value: {bst.find_min(root).val}")
    print(f"Maximum value: {bst.find_max(root).val}")
    print()
    
    # Example 3: BST validation
    print("3. BST Validation:")
    print(f"Is valid BST: {bst.is_valid_bst(root)}")
    print(f"Is valid BST (inorder method): {bst.is_valid_bst_inorder(root)}")
    print()
    
    # Example 4: Kth smallest/largest
    print("4. Kth Smallest/Largest:")
    for k in range(1, 4):
        kth_small = bst.kth_smallest(root, k)
        kth_large = bst.kth_largest(root, k)
        print(f"{k}th smallest: {kth_small}, {k}th largest: {kth_large}")
    print()
    
    # Example 5: Floor and Ceil
    print("5. Floor and Ceil Operations:")
    test_values = [9, 11, 13, 17, 26]
    for val in test_values:
        floor_val = bst.floor_in_bst(root, val)
        ceil_val = bst.ceil_in_bst(root, val)
        print(f"Value {val}: floor = {floor_val}, ceil = {ceil_val}")
    print()
    
    # Example 6: Range operations
    print("6. Range Operations:")
    low, high = 10, 20
    range_values = bst.find_range(root, low, high)
    print(f"Values in range [{low}, {high}]: {range_values}")
    
    range_count = bst.count_nodes_in_range(root, low, high)
    print(f"Count of nodes in range: {range_count}")
    print()
    
    # Example 7: Deletion
    print("7. Node Deletion:")
    print("Before deletion:")
    bst.print_tree_structure(root)
    
    delete_val = 10
    print(f"Deleting {delete_val}")
    root = bst.delete_node(root, delete_val)
    
    print("After deletion:")
    bst.print_tree_structure(root)
    print(f"Inorder after deletion: {bst.inorder_traversal(root)}")
    print()
    
    # Example 8: Conversions
    print("8. BST Conversions:")
    
    # Sorted array to BST
    sorted_array = [1, 3, 5, 7, 9, 11, 13]
    print(f"Sorted array: {sorted_array}")
    balanced_bst = bst.sorted_array_to_bst(sorted_array)
    print("Balanced BST from sorted array:")
    bst.print_tree_structure(balanced_bst)
    
    # BST to sorted array
    bst_array = bst.bst_to_sorted_array(balanced_bst)
    print(f"BST back to sorted array: {bst_array}")
    
    # BST to linked list
    linked_list_head = bst.bst_to_sorted_list(balanced_bst)
    print("BST to linked list:", end=" ")
    current = linked_list_head
    while current:
        print(current.val, end=" -> " if current.next else "\n")
        current = current.next
    print()
    
    # Example 9: LCA in BST
    print("9. Lowest Common Ancestor:")
    # Create nodes for LCA testing
    node8 = TreeNode(8)
    node12 = TreeNode(12)
    
    # Insert these nodes into our BST for testing
    test_root = bst.build_bst_from_list([15, 10, 20, 8, 12, 16, 25])
    
    # Find the actual nodes in the tree
    actual_node8 = bst.search(test_root, 8)
    actual_node12 = bst.search(test_root, 12)
    
    if actual_node8 and actual_node12:
        lca = bst.lowest_common_ancestor_bst(test_root, actual_node8, actual_node12)
        print(f"LCA of {actual_node8.val} and {actual_node12.val}: {lca.val}")
    
    # Example 10: BST trimming
    print("10. BST Trimming:")
    print("Original BST:")
    bst.print_tree_structure(test_root)
    
    trimmed = bst.trim_bst(test_root, 10, 20)
    print(f"Trimmed BST (range [10, 20]):")
    bst.print_tree_structure(trimmed)
    
    print("\n=== Demo Complete ===") 