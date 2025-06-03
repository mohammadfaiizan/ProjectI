"""
Tree Traversal Variants - Advanced Tree Traversal Algorithms
This module implements various tree traversal techniques and specialized traversals.
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

class TreeTraversals:
    
    def __init__(self):
        """Initialize tree traversal algorithms"""
        pass
    
    # ==================== RECURSIVE TRAVERSALS ====================
    
    def inorder_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """Inorder traversal: Left -> Root -> Right (Recursive)"""
        result = []
        
        def inorder(node):
            if node:
                inorder(node.left)
                result.append(node.val)
                inorder(node.right)
        
        inorder(root)
        return result
    
    def preorder_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """Preorder traversal: Root -> Left -> Right (Recursive)"""
        result = []
        
        def preorder(node):
            if node:
                result.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        return result
    
    def postorder_recursive(self, root: Optional[TreeNode]) -> List[int]:
        """Postorder traversal: Left -> Right -> Root (Recursive)"""
        result = []
        
        def postorder(node):
            if node:
                postorder(node.left)
                postorder(node.right)
                result.append(node.val)
        
        postorder(root)
        return result
    
    # ==================== ITERATIVE TRAVERSALS USING STACK ====================
    
    def inorder_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """
        Inorder traversal using stack (Iterative)
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Inorder traversal result
        """
        result = []
        stack = []
        current = root
        
        while stack or current:
            # Go to the leftmost node
            while current:
                stack.append(current)
                current = current.left
            
            # Current must be None here, so we backtrack
            current = stack.pop()
            result.append(current.val)
            
            # Visit right subtree
            current = current.right
        
        return result
    
    def preorder_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """Preorder traversal using stack (Iterative)"""
        if not root:
            return []
        
        result = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            result.append(node.val)
            
            # Push right first so left is processed first
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        
        return result
    
    def postorder_iterative(self, root: Optional[TreeNode]) -> List[int]:
        """Postorder traversal using stack (Iterative)"""
        if not root:
            return []
        
        result = []
        stack = []
        last_visited = None
        current = root
        
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
    
    def postorder_iterative_two_stacks(self, root: Optional[TreeNode]) -> List[int]:
        """Alternative postorder using two stacks"""
        if not root:
            return []
        
        stack1 = [root]
        stack2 = []
        result = []
        
        while stack1:
            node = stack1.pop()
            stack2.append(node)
            
            if node.left:
                stack1.append(node.left)
            if node.right:
                stack1.append(node.right)
        
        while stack2:
            result.append(stack2.pop().val)
        
        return result
    
    # ==================== MORRIS TRAVERSAL (NO STACK/RECURSION) ====================
    
    def morris_inorder(self, root: Optional[TreeNode]) -> List[int]:
        """
        Morris Inorder Traversal - O(1) space complexity
        
        Time Complexity: O(n)
        Space Complexity: O(1)
        
        Uses threading to traverse without stack or recursion
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Inorder traversal result
        """
        result = []
        current = root
        
        while current:
            if not current.left:
                # No left child, visit current and go right
                result.append(current.val)
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Make current the right child of its inorder predecessor
                    predecessor.right = current
                    current = current.left
                else:
                    # Revert the changes: remove the link
                    predecessor.right = None
                    result.append(current.val)
                    current = current.right
        
        return result
    
    def morris_preorder(self, root: Optional[TreeNode]) -> List[int]:
        """Morris Preorder Traversal - O(1) space complexity"""
        result = []
        current = root
        
        while current:
            if not current.left:
                result.append(current.val)
                current = current.right
            else:
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Add to result before going left (preorder)
                    result.append(current.val)
                    predecessor.right = current
                    current = current.left
                else:
                    predecessor.right = None
                    current = current.right
        
        return result
    
    # ==================== LEVEL ORDER TRAVERSALS ====================
    
    def level_order_traversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Level order traversal (BFS) - returns levels separately
        
        Time Complexity: O(n)
        Space Complexity: O(w) where w is maximum width
        
        Args:
            root: Root of the tree
        
        Returns:
            list: List of levels, each level is a list of values
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level_values.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level_values)
        
        return result
    
    def zigzag_level_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Zigzag level order traversal (alternating left-to-right and right-to-left)
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Zigzag level order traversal
        """
        if not root:
            return []
        
        result = []
        queue = deque([root])
        left_to_right = True
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level_values.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            if not left_to_right:
                level_values.reverse()
            
            result.append(level_values)
            left_to_right = not left_to_right
        
        return result
    
    def reverse_level_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """Level order traversal from bottom to top"""
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level_values.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(level_values)
        
        return result[::-1]  # Reverse the levels
    
    # ==================== BOUNDARY TRAVERSALS ====================
    
    def boundary_traversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Boundary traversal: left boundary + leaves + right boundary
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Boundary traversal result
        """
        if not root:
            return []
        
        result = []
        
        # Add root (if it's not a leaf)
        if not self._is_leaf(root):
            result.append(root.val)
        
        # Add left boundary (excluding leaves)
        self._add_left_boundary(root.left, result)
        
        # Add all leaves
        self._add_leaves(root, result)
        
        # Add right boundary (excluding leaves, in reverse)
        self._add_right_boundary(root.right, result)
        
        return result
    
    def _is_leaf(self, node: Optional[TreeNode]) -> bool:
        """Check if node is a leaf"""
        return node and not node.left and not node.right
    
    def _add_left_boundary(self, node: Optional[TreeNode], result: List[int]):
        """Add left boundary nodes (excluding leaves)"""
        if not node or self._is_leaf(node):
            return
        
        result.append(node.val)
        
        if node.left:
            self._add_left_boundary(node.left, result)
        else:
            self._add_left_boundary(node.right, result)
    
    def _add_right_boundary(self, node: Optional[TreeNode], result: List[int]):
        """Add right boundary nodes (excluding leaves, in reverse)"""
        if not node or self._is_leaf(node):
            return
        
        if node.right:
            self._add_right_boundary(node.right, result)
        else:
            self._add_right_boundary(node.left, result)
        
        result.append(node.val)  # Add after recursive call for reverse order
    
    def _add_leaves(self, node: Optional[TreeNode], result: List[int]):
        """Add all leaf nodes"""
        if not node:
            return
        
        if self._is_leaf(node):
            result.append(node.val)
            return
        
        self._add_leaves(node.left, result)
        self._add_leaves(node.right, result)
    
    # ==================== VERTICAL ORDER TRAVERSAL ====================
    
    def vertical_order_traversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Vertical order traversal - nodes at same horizontal distance grouped together
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Vertical order traversal
        """
        if not root:
            return []
        
        # Dictionary to store nodes at each horizontal distance
        vertical_map = defaultdict(list)
        
        # Queue for BFS: (node, horizontal_distance, level)
        queue = deque([(root, 0, 0)])
        
        while queue:
            node, hd, level = queue.popleft()
            vertical_map[hd].append((level, node.val))
            
            if node.left:
                queue.append((node.left, hd - 1, level + 1))
            if node.right:
                queue.append((node.right, hd + 1, level + 1))
        
        # Sort by horizontal distance and prepare result
        result = []
        for hd in sorted(vertical_map.keys()):
            # Sort by level, then by value for same level
            vertical_map[hd].sort()
            result.append([val for level, val in vertical_map[hd]])
        
        return result
    
    def vertical_order_traversal_with_coordinates(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Vertical order traversal with proper coordinate-based sorting
        """
        if not root:
            return []
        
        # List to store (x, y, val) coordinates
        coordinates = []
        
        def dfs(node, x, y):
            if node:
                coordinates.append((x, y, node.val))
                dfs(node.left, x - 1, y + 1)
                dfs(node.right, x + 1, y + 1)
        
        dfs(root, 0, 0)
        
        # Sort by x-coordinate, then by y-coordinate, then by value
        coordinates.sort()
        
        # Group by x-coordinate
        result = []
        i = 0
        while i < len(coordinates):
            column = []
            current_x = coordinates[i][0]
            
            while i < len(coordinates) and coordinates[i][0] == current_x:
                column.append(coordinates[i][2])
                i += 1
            
            result.append(column)
        
        return result
    
    # ==================== DIAGONAL TRAVERSAL ====================
    
    def diagonal_traversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Diagonal traversal - nodes at same diagonal grouped together
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Diagonal traversal result
        """
        if not root:
            return []
        
        diagonal_map = defaultdict(list)
        
        def dfs(node, diagonal):
            if not node:
                return
            
            diagonal_map[diagonal].append(node.val)
            
            # Left child increases diagonal distance
            dfs(node.left, diagonal + 1)
            # Right child stays at same diagonal
            dfs(node.right, diagonal)
        
        dfs(root, 0)
        
        # Convert to list format
        result = []
        for d in sorted(diagonal_map.keys()):
            result.append(diagonal_map[d])
        
        return result
    
    def anti_diagonal_traversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """Anti-diagonal traversal (slope = +1)"""
        if not root:
            return []
        
        diagonal_map = defaultdict(list)
        
        def dfs(node, diagonal):
            if not node:
                return
            
            diagonal_map[diagonal].append(node.val)
            
            # Left child stays at same anti-diagonal
            dfs(node.left, diagonal)
            # Right child increases anti-diagonal distance
            dfs(node.right, diagonal + 1)
        
        dfs(root, 0)
        
        result = []
        for d in sorted(diagonal_map.keys()):
            result.append(diagonal_map[d])
        
        return result
    
    # ==================== TOP VIEW / BOTTOM VIEW ====================
    
    def top_view(self, root: Optional[TreeNode]) -> List[int]:
        """
        Top view of binary tree - first node at each horizontal distance
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Top view traversal
        """
        if not root:
            return []
        
        # Dictionary to store first node at each horizontal distance
        top_map = {}
        
        # Queue for BFS: (node, horizontal_distance)
        queue = deque([(root, 0)])
        
        while queue:
            node, hd = queue.popleft()
            
            # If this horizontal distance is seen for first time
            if hd not in top_map:
                top_map[hd] = node.val
            
            if node.left:
                queue.append((node.left, hd - 1))
            if node.right:
                queue.append((node.right, hd + 1))
        
        # Sort by horizontal distance and return values
        return [top_map[hd] for hd in sorted(top_map.keys())]
    
    def bottom_view(self, root: Optional[TreeNode]) -> List[int]:
        """
        Bottom view of binary tree - last node at each horizontal distance
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Bottom view traversal
        """
        if not root:
            return []
        
        # Dictionary to store last node at each horizontal distance
        bottom_map = {}
        
        # Queue for BFS: (node, horizontal_distance)
        queue = deque([(root, 0)])
        
        while queue:
            node, hd = queue.popleft()
            
            # Update with latest node at this horizontal distance
            bottom_map[hd] = node.val
            
            if node.left:
                queue.append((node.left, hd - 1))
            if node.right:
                queue.append((node.right, hd + 1))
        
        # Sort by horizontal distance and return values
        return [bottom_map[hd] for hd in sorted(bottom_map.keys())]
    
    def left_view(self, root: Optional[TreeNode]) -> List[int]:
        """
        Left view of binary tree - leftmost node at each level
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Left view traversal
        """
        if not root:
            return []
        
        result = []
        
        def dfs(node, level):
            if not node:
                return
            
            # If this is the first node at this level
            if level == len(result):
                result.append(node.val)
            
            dfs(node.left, level + 1)
            dfs(node.right, level + 1)
        
        dfs(root, 0)
        return result
    
    def right_view(self, root: Optional[TreeNode]) -> List[int]:
        """
        Right view of binary tree - rightmost node at each level
        
        Args:
            root: Root of the tree
        
        Returns:
            list: Right view traversal
        """
        if not root:
            return []
        
        result = []
        
        def dfs(node, level):
            if not node:
                return
            
            # If this is the first node at this level (processing right first)
            if level == len(result):
                result.append(node.val)
            
            dfs(node.right, level + 1)
            dfs(node.left, level + 1)
        
        dfs(root, 0)
        return result
    
    # ==================== UTILITY METHODS ====================
    
    def build_tree_from_array(self, arr: List[Optional[int]]) -> Optional[TreeNode]:
        """Build tree from array representation for testing"""
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
        """Print tree structure for visualization"""
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
    print("=== Tree Traversal Variants Demo ===\n")
    
    traversals = TreeTraversals()
    
    # Build test tree: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    root = traversals.build_tree_from_array(arr)
    
    print("Test Tree Structure:")
    traversals.print_tree_structure(root)
    print()
    
    # Example 1: Basic traversals (recursive vs iterative)
    print("1. Basic Traversals (Recursive vs Iterative):")
    print(f"Inorder (Recursive):  {traversals.inorder_recursive(root)}")
    print(f"Inorder (Iterative):  {traversals.inorder_iterative(root)}")
    print(f"Preorder (Recursive): {traversals.preorder_recursive(root)}")
    print(f"Preorder (Iterative): {traversals.preorder_iterative(root)}")
    print(f"Postorder (Recursive):{traversals.postorder_recursive(root)}")
    print(f"Postorder (Iterative):{traversals.postorder_iterative(root)}")
    print()
    
    # Example 2: Morris Traversals
    print("2. Morris Traversals (O(1) Space):")
    print(f"Morris Inorder:  {traversals.morris_inorder(root)}")
    print(f"Morris Preorder: {traversals.morris_preorder(root)}")
    print()
    
    # Example 3: Level order variations
    print("3. Level Order Variations:")
    level_order = traversals.level_order_traversal(root)
    print(f"Level Order: {level_order}")
    
    zigzag_order = traversals.zigzag_level_order(root)
    print(f"Zigzag Order: {zigzag_order}")
    
    reverse_level = traversals.reverse_level_order(root)
    print(f"Reverse Level Order: {reverse_level}")
    print()
    
    # Example 4: Boundary traversal
    print("4. Boundary Traversal:")
    boundary = traversals.boundary_traversal(root)
    print(f"Boundary: {boundary}")
    print()
    
    # Example 5: Vertical order traversal
    print("5. Vertical Order Traversal:")
    vertical = traversals.vertical_order_traversal(root)
    print(f"Vertical Order: {vertical}")
    
    vertical_coord = traversals.vertical_order_traversal_with_coordinates(root)
    print(f"Vertical (Coordinate-based): {vertical_coord}")
    print()
    
    # Example 6: Diagonal traversals
    print("6. Diagonal Traversals:")
    diagonal = traversals.diagonal_traversal(root)
    print(f"Diagonal: {diagonal}")
    
    anti_diagonal = traversals.anti_diagonal_traversal(root)
    print(f"Anti-diagonal: {anti_diagonal}")
    print()
    
    # Example 7: View traversals
    print("7. View Traversals:")
    top_view = traversals.top_view(root)
    print(f"Top View: {top_view}")
    
    bottom_view = traversals.bottom_view(root)
    print(f"Bottom View: {bottom_view}")
    
    left_view = traversals.left_view(root)
    print(f"Left View: {left_view}")
    
    right_view = traversals.right_view(root)
    print(f"Right View: {right_view}")
    print()
    
    # Example 8: Test with a different tree structure
    print("8. Test with Skewed Tree:")
    skewed_arr = [1, 2, None, 3, None, None, None, 4]
    skewed_root = traversals.build_tree_from_array(skewed_arr)
    
    print("Skewed Tree Structure:")
    traversals.print_tree_structure(skewed_root)
    
    print(f"Skewed Top View: {traversals.top_view(skewed_root)}")
    print(f"Skewed Bottom View: {traversals.bottom_view(skewed_root)}")
    print(f"Skewed Left View: {traversals.left_view(skewed_root)}")
    print(f"Skewed Right View: {traversals.right_view(skewed_root)}")
    
    print("\n=== Demo Complete ===") 