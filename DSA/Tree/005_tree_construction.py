"""
Tree Construction - Building Trees from Various Inputs
This module implements algorithms to construct binary trees from different representations.
"""

from collections import deque, defaultdict
from typing import List, Optional, Dict, Tuple

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeConstruction:
    
    def __init__(self):
        """Initialize tree construction algorithms"""
        pass
    
    # ==================== CONSTRUCTION FROM TRAVERSALS ====================
    
    def build_from_inorder_preorder(self, inorder: List[int], preorder: List[int]) -> Optional[TreeNode]:
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
    
    def build_from_inorder_postorder(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
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
    
    def build_from_preorder_postorder(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        """
        Build binary tree from preorder and postorder (one of many possible trees)
        
        Args:
            preorder: Preorder traversal
            postorder: Postorder traversal
        
        Returns:
            TreeNode: Root of one possible tree
        """
        if not preorder or not postorder:
            return None
        
        def build_tree(pre_start, pre_end, post_start, post_end):
            if pre_start > pre_end:
                return None
            
            root = TreeNode(preorder[pre_start])
            
            if pre_start == pre_end:
                return root
            
            # Find left subtree root in postorder
            left_root = preorder[pre_start + 1]
            left_root_idx = -1
            
            for i in range(post_start, post_end + 1):
                if postorder[i] == left_root:
                    left_root_idx = i
                    break
            
            if left_root_idx == -1:
                return root
            
            # Calculate sizes
            left_size = left_root_idx - post_start + 1
            
            # Build subtrees
            root.left = build_tree(pre_start + 1, pre_start + left_size, 
                                 post_start, left_root_idx)
            root.right = build_tree(pre_start + left_size + 1, pre_end,
                                  left_root_idx + 1, post_end - 1)
            
            return root
        
        return build_tree(0, len(preorder) - 1, 0, len(postorder) - 1)
    
    def build_from_inorder_levelorder(self, inorder: List[int], levelorder: List[int]) -> Optional[TreeNode]:
        """
        Build binary tree from inorder and level order traversals
        
        Args:
            inorder: Inorder traversal
            levelorder: Level order traversal
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not inorder or not levelorder:
            return None
        
        # Create set for quick lookup
        inorder_set = set(inorder)
        
        def build_tree(level_nodes, in_start, in_end):
            if in_start > in_end or not level_nodes:
                return None
            
            # Find root (first element in level order that's in current inorder range)
            root_val = None
            for node in level_nodes:
                if node in inorder_set:
                    # Check if this node is in current inorder range
                    root_idx = inorder.index(node)
                    if in_start <= root_idx <= in_end:
                        root_val = node
                        break
            
            if root_val is None:
                return None
            
            root = TreeNode(root_val)
            root_idx = inorder.index(root_val)
            
            # Filter level order for left and right subtrees
            left_nodes = []
            right_nodes = []
            
            for node in level_nodes:
                if node == root_val:
                    continue
                node_idx = -1
                try:
                    node_idx = inorder.index(node)
                except ValueError:
                    continue
                
                if in_start <= node_idx < root_idx:
                    left_nodes.append(node)
                elif root_idx < node_idx <= in_end:
                    right_nodes.append(node)
            
            # Build subtrees
            root.left = build_tree(left_nodes, in_start, root_idx - 1)
            root.right = build_tree(right_nodes, root_idx + 1, in_end)
            
            return root
        
        return build_tree(levelorder, 0, len(inorder) - 1)
    
    # ==================== CONSTRUCTION FROM ARRAYS ====================
    
    def build_from_array_levelorder(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
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
    
    def build_from_parent_array(self, parent: List[int]) -> Optional[TreeNode]:
        """
        Build binary tree from parent array
        parent[i] is the parent of node i, -1 for root
        
        Args:
            parent: Parent array
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not parent:
            return None
        
        n = len(parent)
        nodes = [TreeNode(i) for i in range(n)]
        root = None
        
        for i in range(n):
            if parent[i] == -1:
                root = nodes[i]
            else:
                parent_node = nodes[parent[i]]
                if not parent_node.left:
                    parent_node.left = nodes[i]
                else:
                    parent_node.right = nodes[i]
        
        return root
    
    def build_from_parent_array_with_positions(self, parent: List[int], positions: List[str]) -> Optional[TreeNode]:
        """
        Build binary tree from parent array with position indicators
        
        Args:
            parent: Parent array
            positions: Position array ('L' for left, 'R' for right, 'N' for root)
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not parent or not positions or len(parent) != len(positions):
            return None
        
        n = len(parent)
        nodes = [TreeNode(i) for i in range(n)]
        root = None
        
        for i in range(n):
            if positions[i] == 'N':  # Root
                root = nodes[i]
            elif positions[i] == 'L':  # Left child
                if parent[i] != -1:
                    nodes[parent[i]].left = nodes[i]
            elif positions[i] == 'R':  # Right child
                if parent[i] != -1:
                    nodes[parent[i]].right = nodes[i]
        
        return root
    
    # ==================== CONSTRUCTION FROM STRINGS ====================
    
    def build_from_string_brackets(self, data: str) -> Optional[TreeNode]:
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
    
    def build_from_newick_format(self, newick: str) -> Optional[TreeNode]:
        """
        Build tree from Newick format (phylogenetic tree format)
        Example: "((A,B)C,D)E;"
        
        Args:
            newick: Newick format string
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not newick:
            return None
        
        stack = []
        i = 0
        node_id = 0
        
        while i < len(newick):
            if newick[i] == '(':
                # Start of new subtree
                node = TreeNode(f"internal_{node_id}")
                node_id += 1
                if stack:
                    parent = stack[-1]
                    if not parent.left:
                        parent.left = node
                    else:
                        parent.right = node
                stack.append(node)
                i += 1
            elif newick[i] == ')':
                # End of subtree
                if stack:
                    stack.pop()
                i += 1
            elif newick[i] == ',':
                # Separator
                i += 1
            elif newick[i] == ';':
                # End of tree
                break
            else:
                # Leaf node or internal node name
                start = i
                while i < len(newick) and newick[i] not in '(),;':
                    i += 1
                
                name = newick[start:i]
                if name:
                    if stack:
                        # This is a name for the current internal node
                        stack[-1].val = name
                    else:
                        # This is a leaf node
                        node = TreeNode(name)
                        if stack:
                            parent = stack[-1]
                            if not parent.left:
                                parent.left = node
                            else:
                                parent.right = node
        
        return stack[0] if stack else None
    
    def build_from_preorder_markers(self, preorder: List[str]) -> Optional[TreeNode]:
        """
        Build tree from preorder traversal with None markers
        
        Args:
            preorder: Preorder traversal with None markers
        
        Returns:
            TreeNode: Root of constructed tree
        """
        if not preorder:
            return None
        
        self.index = 0
        
        def build():
            if self.index >= len(preorder) or preorder[self.index] is None:
                self.index += 1
                return None
            
            root = TreeNode(int(preorder[self.index]))
            self.index += 1
            
            root.left = build()
            root.right = build()
            
            return root
        
        return build()
    
    # ==================== SPECIAL TREE CONSTRUCTIONS ====================
    
    def build_complete_binary_tree(self, n: int) -> Optional[TreeNode]:
        """
        Build complete binary tree with n nodes (values 1 to n)
        
        Args:
            n: Number of nodes
        
        Returns:
            TreeNode: Root of complete binary tree
        """
        if n <= 0:
            return None
        
        def build_tree(index):
            if index > n:
                return None
            
            root = TreeNode(index)
            root.left = build_tree(2 * index)
            root.right = build_tree(2 * index + 1)
            
            return root
        
        return build_tree(1)
    
    def build_perfect_binary_tree(self, height: int) -> Optional[TreeNode]:
        """
        Build perfect binary tree of given height
        
        Args:
            height: Height of tree (0 for single node)
        
        Returns:
            TreeNode: Root of perfect binary tree
        """
        if height < 0:
            return None
        
        def build_tree(current_height, value):
            if current_height > height:
                return None
            
            root = TreeNode(value[0])
            value[0] += 1
            
            if current_height < height:
                root.left = build_tree(current_height + 1, value)
                root.right = build_tree(current_height + 1, value)
            
            return root
        
        return build_tree(0, [1])
    
    def build_from_sorted_array(self, nums: List[int]) -> Optional[TreeNode]:
        """
        Convert sorted array to balanced BST
        
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
    
    def build_from_linked_list(self, head) -> Optional[TreeNode]:
        """
        Build height-balanced BST from sorted linked list
        
        Args:
            head: Head of sorted linked list
        
        Returns:
            TreeNode: Root of balanced BST
        """
        def get_length(node):
            length = 0
            while node:
                length += 1
                node = node.next
            return length
        
        def build_tree(length):
            if length <= 0:
                return None
            
            left = build_tree(length // 2)
            
            root = TreeNode(self.current.val)
            self.current = self.current.next
            
            right = build_tree(length - length // 2 - 1)
            
            root.left = left
            root.right = right
            
            return root
        
        self.current = head
        length = get_length(head)
        return build_tree(length)
    
    # ==================== UTILITY METHODS ====================
    
    def tree_to_array(self, root: Optional[TreeNode]) -> List[Optional[int]]:
        """Convert tree to array representation"""
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
    
    def tree_to_string_brackets(self, root: Optional[TreeNode]) -> str:
        """Convert tree to string representation with brackets"""
        if not root:
            return ""
        
        result = str(root.val)
        
        if root.left or root.right:
            result += "("
            if root.left:
                result += self.tree_to_string_brackets(root.left)
            result += ")"
            
            if root.right:
                result += "("
                result += self.tree_to_string_brackets(root.right)
                result += ")"
        
        return result
    
    def inorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """Get inorder traversal for verification"""
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    def preorder_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """Get preorder traversal for verification"""
        result = []
        
        def preorder(node):
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        return result
    
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
    print("=== Tree Construction Demo ===\n")
    
    construction = TreeConstruction()
    
    # Example 1: Construction from traversals
    print("1. Construction from Traversals:")
    inorder = [4, 2, 5, 1, 6, 3, 7]
    preorder = [1, 2, 4, 5, 3, 6, 7]
    postorder = [4, 5, 2, 6, 7, 3, 1]
    levelorder = [1, 2, 3, 4, 5, 6, 7]
    
    print(f"Inorder:  {inorder}")
    print(f"Preorder: {preorder}")
    print(f"Postorder: {postorder}")
    print(f"Levelorder: {levelorder}")
    print()
    
    # Build from inorder + preorder
    tree1 = construction.build_from_inorder_preorder(inorder, preorder)
    print("Tree from inorder + preorder:")
    construction.print_tree_structure(tree1)
    print(f"Verification - Inorder: {construction.inorder_traversal(tree1)}")
    print(f"Verification - Preorder: {construction.preorder_traversal(tree1)}")
    print()
    
    # Build from inorder + postorder
    tree2 = construction.build_from_inorder_postorder(inorder, postorder)
    print("Tree from inorder + postorder:")
    construction.print_tree_structure(tree2)
    print()
    
    # Build from inorder + levelorder
    tree3 = construction.build_from_inorder_levelorder(inorder, levelorder)
    print("Tree from inorder + levelorder:")
    construction.print_tree_structure(tree3)
    print()
    
    # Example 2: Construction from arrays
    print("2. Construction from Arrays:")
    
    # Array representation
    arr = [1, 2, 3, 4, 5, None, 6, 7, 8]
    print(f"Array: {arr}")
    tree_from_array = construction.build_from_array_levelorder(arr)
    print("Tree from array:")
    construction.print_tree_structure(tree_from_array)
    
    # Parent array
    parent_arr = [-1, 0, 0, 1, 1, 2, 2]
    print(f"Parent array: {parent_arr}")
    tree_from_parent = construction.build_from_parent_array(parent_arr)
    print("Tree from parent array:")
    construction.print_tree_structure(tree_from_parent)
    print()
    
    # Example 3: Construction from strings
    print("3. Construction from Strings:")
    
    # Bracket notation
    bracket_string = "1(2(4)(5))(3(6)(7))"
    print(f"Bracket string: {bracket_string}")
    tree_from_brackets = construction.build_from_string_brackets(bracket_string)
    print("Tree from bracket string:")
    construction.print_tree_structure(tree_from_brackets)
    
    # Convert back to string
    back_to_string = construction.tree_to_string_brackets(tree_from_brackets)
    print(f"Back to string: {back_to_string}")
    print()
    
    # Preorder with markers
    preorder_markers = ['1', '2', '4', None, None, '5', None, None, '3', '6', None, None, '7', None, None]
    print(f"Preorder with markers: {preorder_markers}")
    tree_from_markers = construction.build_from_preorder_markers(preorder_markers)
    print("Tree from preorder markers:")
    construction.print_tree_structure(tree_from_markers)
    print()
    
    # Example 4: Special constructions
    print("4. Special Tree Constructions:")
    
    # Complete binary tree
    complete_tree = construction.build_complete_binary_tree(7)
    print("Complete binary tree (7 nodes):")
    construction.print_tree_structure(complete_tree)
    
    # Perfect binary tree
    perfect_tree = construction.build_perfect_binary_tree(2)
    print("Perfect binary tree (height 2):")
    construction.print_tree_structure(perfect_tree)
    
    # Balanced BST from sorted array
    sorted_arr = [1, 3, 5, 7, 9, 11, 13]
    print(f"Sorted array: {sorted_arr}")
    bst_tree = construction.build_from_sorted_array(sorted_arr)
    print("Balanced BST from sorted array:")
    construction.print_tree_structure(bst_tree)
    print()
    
    # Example 5: Verification
    print("5. Construction Verification:")
    test_tree = construction.build_from_array_levelorder([1, 2, 3, 4, 5, 6, 7])
    
    original_array = construction.tree_to_array(test_tree)
    inorder_result = construction.inorder_traversal(test_tree)
    preorder_result = construction.preorder_traversal(test_tree)
    
    print("Original tree:")
    construction.print_tree_structure(test_tree)
    print(f"Array representation: {original_array}")
    print(f"Inorder traversal: {inorder_result}")
    print(f"Preorder traversal: {preorder_result}")
    
    print("\n=== Demo Complete ===") 