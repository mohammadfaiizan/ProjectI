"""
Tree Properties & Algorithms - Advanced Tree Property Analysis
This module implements algorithms for analyzing various tree properties and characteristics.
"""

from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict, Set
import hashlib

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class TreeProperties:
    
    def __init__(self):
        """Initialize tree properties analyzer"""
        pass
    
    # ==================== TREE DIAMETER ====================
    
    def diameter_two_pass(self, root: Optional[TreeNode]) -> int:
        """
        Calculate tree diameter using two-pass approach
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Diameter of the tree
        """
        if not root:
            return 0
        
        self.max_diameter = 0
        
        def calculate_height(node):
            if not node:
                return 0
            
            left_height = calculate_height(node.left)
            right_height = calculate_height(node.right)
            
            # Update diameter at this node
            current_diameter = left_height + right_height
            self.max_diameter = max(self.max_diameter, current_diameter)
            
            return max(left_height, right_height) + 1
        
        calculate_height(root)
        return self.max_diameter
    
    def diameter_optimized(self, root: Optional[TreeNode]) -> int:
        """
        Calculate tree diameter with optimized single pass
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Diameter of the tree
        """
        def diameter_helper(node):
            if not node:
                return 0, 0  # height, diameter
            
            left_height, left_diameter = diameter_helper(node.left)
            right_height, right_diameter = diameter_helper(node.right)
            
            current_height = max(left_height, right_height) + 1
            current_diameter = max(
                left_diameter,
                right_diameter,
                left_height + right_height
            )
            
            return current_height, current_diameter
        
        if not root:
            return 0
        
        _, diameter = diameter_helper(root)
        return diameter
    
    def diameter_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Calculate diameter and return the path that forms the diameter
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (diameter, path)
        """
        if not root:
            return 0, []
        
        self.max_diameter = 0
        self.diameter_path = []
        
        def find_diameter_path(node):
            if not node:
                return 0, []
            
            left_height, left_path = find_diameter_path(node.left)
            right_height, right_path = find_diameter_path(node.right)
            
            current_diameter = left_height + right_height
            
            if current_diameter > self.max_diameter:
                self.max_diameter = current_diameter
                # Construct path: left_path (reversed) + node + right_path
                self.diameter_path = (
                    left_path[::-1] + [node.val] + right_path
                )
            
            # Return height and path to deepest node
            if left_height > right_height:
                return left_height + 1, left_path + [node.val]
            else:
                return right_height + 1, right_path + [node.val]
        
        find_diameter_path(root)
        return self.max_diameter, self.diameter_path
    
    # ==================== MAXIMUM WIDTH OF BINARY TREE ====================
    
    def max_width_level_order(self, root: Optional[TreeNode]) -> int:
        """
        Find maximum width using level order traversal
        
        Time Complexity: O(n)
        Space Complexity: O(w) where w is maximum width
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Maximum width of tree
        """
        if not root:
            return 0
        
        max_width = 0
        queue = deque([(root, 0)])  # (node, position)
        
        while queue:
            level_size = len(queue)
            level_min = queue[0][1]
            level_max = queue[0][1]
            
            for _ in range(level_size):
                node, pos = queue.popleft()
                level_max = pos
                
                if node.left:
                    queue.append((node.left, 2 * pos))
                if node.right:
                    queue.append((node.right, 2 * pos + 1))
            
            max_width = max(max_width, level_max - level_min + 1)
        
        return max_width
    
    def max_width_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Find maximum width using DFS approach
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            int: Maximum width of tree
        """
        if not root:
            return 0
        
        level_min = {}
        level_max = {}
        
        def dfs(node, level, pos):
            if not node:
                return
            
            if level not in level_min:
                level_min[level] = pos
                level_max[level] = pos
            else:
                level_min[level] = min(level_min[level], pos)
                level_max[level] = max(level_max[level], pos)
            
            dfs(node.left, level + 1, 2 * pos)
            dfs(node.right, level + 1, 2 * pos + 1)
        
        dfs(root, 0, 0)
        
        max_width = 0
        for level in level_min:
            width = level_max[level] - level_min[level] + 1
            max_width = max(max_width, width)
        
        return max_width
    
    def max_width_with_details(self, root: Optional[TreeNode]) -> Tuple[int, int, List[int]]:
        """
        Find maximum width with level details
        
        Args:
            root: Root of binary tree
        
        Returns:
            Tuple of (max_width, level_with_max_width, values_at_max_level)
        """
        if not root:
            return 0, -1, []
        
        level_data = {}  # level -> list of (position, value)
        
        def dfs(node, level, pos):
            if not node:
                return
            
            if level not in level_data:
                level_data[level] = []
            
            level_data[level].append((pos, node.val))
            
            dfs(node.left, level + 1, 2 * pos)
            dfs(node.right, level + 1, 2 * pos + 1)
        
        dfs(root, 0, 0)
        
        max_width = 0
        max_level = -1
        max_level_values = []
        
        for level, positions in level_data.items():
            positions.sort()  # Sort by position
            if positions:
                width = positions[-1][0] - positions[0][0] + 1
                if width > max_width:
                    max_width = width
                    max_level = level
                    max_level_values = [val for _, val in positions]
        
        return max_width, max_level, max_level_values
    
    # ==================== SYMMETRIC TREE ====================
    
    def is_symmetric_recursive(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is symmetric using recursive approach
        
        Time Complexity: O(n)
        Space Complexity: O(h)
        
        Args:
            root: Root of binary tree
        
        Returns:
            bool: True if tree is symmetric
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
    
    def is_symmetric_iterative(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is symmetric using iterative approach
        
        Time Complexity: O(n)
        Space Complexity: O(w) where w is maximum width
        
        Args:
            root: Root of binary tree
        
        Returns:
            bool: True if tree is symmetric
        """
        if not root:
            return True
        
        queue = deque([(root.left, root.right)])
        
        while queue:
            left, right = queue.popleft()
            
            if not left and not right:
                continue
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            
            queue.append((left.left, right.right))
            queue.append((left.right, right.left))
        
        return True
    
    def is_symmetric_level_check(self, root: Optional[TreeNode]) -> bool:
        """
        Check symmetry by comparing each level
        
        Args:
            root: Root of binary tree
        
        Returns:
            bool: True if tree is symmetric
        """
        if not root:
            return True
        
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                
                if node:
                    level_values.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)
                else:
                    level_values.append(None)
            
            # Check if level is palindromic
            if level_values != level_values[::-1]:
                return False
        
        return True
    
    # ==================== SUBTREE OF ANOTHER TREE ====================
    
    def is_subtree_naive(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        Check if subRoot is subtree of root (naive approach)
        
        Time Complexity: O(m * n) where m, n are sizes of trees
        Space Complexity: O(max(h1, h2))
        
        Args:
            root: Root of main tree
            subRoot: Root of potential subtree
        
        Returns:
            bool: True if subRoot is subtree of root
        """
        def is_same_tree(p, q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            return (p.val == q.val and
                    is_same_tree(p.left, q.left) and
                    is_same_tree(p.right, q.right))
        
        if not root:
            return not subRoot
        
        return (is_same_tree(root, subRoot) or
                self.is_subtree_naive(root.left, subRoot) or
                self.is_subtree_naive(root.right, subRoot))
    
    def is_subtree_serialization(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        Check subtree using serialization approach
        
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        
        Args:
            root: Root of main tree
            subRoot: Root of potential subtree
        
        Returns:
            bool: True if subRoot is subtree of root
        """
        def serialize(node):
            if not node:
                return "null"
            return f"#{node.val}#{serialize(node.left)}#{serialize(node.right)}"
        
        root_serial = serialize(root)
        sub_serial = serialize(subRoot)
        
        return sub_serial in root_serial
    
    def is_subtree_hash(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        """
        Check subtree using hash-based approach
        
        Time Complexity: O(m + n)
        Space Complexity: O(m + n)
        
        Args:
            root: Root of main tree
            subRoot: Root of potential subtree
        
        Returns:
            bool: True if subRoot is subtree of root
        """
        def get_hash(node):
            if not node:
                return "null"
            
            left_hash = get_hash(node.left)
            right_hash = get_hash(node.right)
            
            node_repr = f"{node.val},{left_hash},{right_hash}"
            return hashlib.md5(node_repr.encode()).hexdigest()
        
        if not subRoot:
            return True
        if not root:
            return False
        
        target_hash = get_hash(subRoot)
        
        def find_subtree(node):
            if not node:
                return False
            
            if get_hash(node) == target_hash:
                return True
            
            return find_subtree(node.left) or find_subtree(node.right)
        
        return find_subtree(root)
    
    # ==================== TREE ISOMORPHISM ====================
    
    def is_isomorphic_recursive(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """
        Check if two trees are isomorphic (recursive)
        
        Time Complexity: O(n * m)
        Space Complexity: O(h)
        
        Args:
            root1, root2: Roots of two trees
        
        Returns:
            bool: True if trees are isomorphic
        """
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        if root1.val != root2.val:
            return False
        
        # Check both possibilities: no flip and flip
        return ((self.is_isomorphic_recursive(root1.left, root2.left) and
                 self.is_isomorphic_recursive(root1.right, root2.right)) or
                (self.is_isomorphic_recursive(root1.left, root2.right) and
                 self.is_isomorphic_recursive(root1.right, root2.left)))
    
    def is_isomorphic_canonical(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        """
        Check isomorphism using canonical form
        
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        Args:
            root1, root2: Roots of two trees
        
        Returns:
            bool: True if trees are isomorphic
        """
        def get_canonical_form(node):
            if not node:
                return "null"
            
            left_form = get_canonical_form(node.left)
            right_form = get_canonical_form(node.right)
            
            # Sort children to get canonical form
            children = sorted([left_form, right_form])
            return f"({node.val},{children[0]},{children[1]})"
        
        return get_canonical_form(root1) == get_canonical_form(root2)
    
    def get_isomorphism_classes(self, roots: List[Optional[TreeNode]]) -> Dict[str, List[int]]:
        """
        Group trees by isomorphism classes
        
        Args:
            roots: List of tree roots
        
        Returns:
            Dict mapping canonical forms to tree indices
        """
        def get_canonical_form(node):
            if not node:
                return "null"
            
            left_form = get_canonical_form(node.left)
            right_form = get_canonical_form(node.right)
            
            children = sorted([left_form, right_form])
            return f"({node.val},{children[0]},{children[1]})"
        
        classes = defaultdict(list)
        
        for i, root in enumerate(roots):
            canonical = get_canonical_form(root)
            classes[canonical].append(i)
        
        return dict(classes)
    
    # ==================== ADDITIONAL TREE PROPERTIES ====================
    
    def is_complete_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is complete
        
        Time Complexity: O(n)
        Space Complexity: O(w)
        
        Args:
            root: Root of binary tree
        
        Returns:
            bool: True if tree is complete
        """
        if not root:
            return True
        
        queue = deque([root])
        found_null = False
        
        while queue:
            node = queue.popleft()
            
            if not node:
                found_null = True
            else:
                if found_null:
                    return False
                queue.append(node.left)
                queue.append(node.right)
        
        return True
    
    def is_perfect_tree(self, root: Optional[TreeNode]) -> bool:
        """
        Check if tree is perfect
        
        Args:
            root: Root of binary tree
        
        Returns:
            bool: True if tree is perfect
        """
        def get_height(node):
            if not node:
                return 0
            return max(get_height(node.left), get_height(node.right)) + 1
        
        def is_perfect_helper(node, height, level):
            if not node:
                return level == height
            
            if not node.left and not node.right:
                return level == height
            
            if not node.left or not node.right:
                return False
            
            return (is_perfect_helper(node.left, height, level + 1) and
                    is_perfect_helper(node.right, height, level + 1))
        
        if not root:
            return True
        
        height = get_height(root)
        return is_perfect_helper(root, height, 1)
    
    def get_tree_statistics(self, root: Optional[TreeNode]) -> Dict[str, any]:
        """
        Get comprehensive tree statistics
        
        Args:
            root: Root of binary tree
        
        Returns:
            Dict with various tree statistics
        """
        if not root:
            return {
                'nodes': 0, 'height': 0, 'diameter': 0,
                'leaves': 0, 'internal_nodes': 0, 'max_width': 0,
                'is_complete': True, 'is_perfect': True, 'is_symmetric': True
            }
        
        stats = {}
        
        # Basic counts
        def count_nodes(node):
            if not node:
                return 0, 0  # total_nodes, leaf_nodes
            
            if not node.left and not node.right:
                return 1, 1
            
            left_total, left_leaves = count_nodes(node.left)
            right_total, right_leaves = count_nodes(node.right)
            
            return left_total + right_total + 1, left_leaves + right_leaves
        
        total_nodes, leaf_nodes = count_nodes(root)
        stats['nodes'] = total_nodes
        stats['leaves'] = leaf_nodes
        stats['internal_nodes'] = total_nodes - leaf_nodes
        
        # Height and diameter
        stats['height'] = self.get_height(root)
        stats['diameter'] = self.diameter_optimized(root)
        
        # Width
        stats['max_width'] = self.max_width_level_order(root)
        
        # Properties
        stats['is_complete'] = self.is_complete_tree(root)
        stats['is_perfect'] = self.is_perfect_tree(root)
        stats['is_symmetric'] = self.is_symmetric_recursive(root)
        
        return stats
    
    # ==================== UTILITY METHODS ====================
    
    def get_height(self, root: Optional[TreeNode]) -> int:
        """Get height of tree"""
        if not root:
            return 0
        return max(self.get_height(root.left), self.get_height(root.right)) + 1
    
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
    print("=== Tree Properties Demo ===\n")
    
    analyzer = TreeProperties()
    
    # Example 1: Tree Diameter
    print("1. Tree Diameter Analysis:")
    
    # Test tree for diameter
    diameter_arr = [1, 2, 3, 4, 5, None, None, 6, 7]
    diameter_tree = analyzer.build_tree_from_array(diameter_arr)
    
    print("Diameter test tree:")
    analyzer.print_tree_structure(diameter_tree)
    
    diameter1 = analyzer.diameter_two_pass(diameter_tree)
    diameter2 = analyzer.diameter_optimized(diameter_tree)
    diameter3, path = analyzer.diameter_with_path(diameter_tree)
    
    print(f"Diameter (two-pass): {diameter1}")
    print(f"Diameter (optimized): {diameter2}")
    print(f"Diameter with path: {diameter3}, Path: {path}")
    print()
    
    # Example 2: Maximum Width
    print("2. Maximum Width Analysis:")
    
    width_arr = [1, 3, 2, 5, 3, None, 9]
    width_tree = analyzer.build_tree_from_array(width_arr)
    
    print("Width test tree:")
    analyzer.print_tree_structure(width_tree)
    
    max_width1 = analyzer.max_width_level_order(width_tree)
    max_width2 = analyzer.max_width_dfs(width_tree)
    max_width3, level, values = analyzer.max_width_with_details(width_tree)
    
    print(f"Max width (level order): {max_width1}")
    print(f"Max width (DFS): {max_width2}")
    print(f"Max width details: width={max_width3}, level={level}, values={values}")
    print()
    
    # Example 3: Symmetric Tree
    print("3. Symmetric Tree Analysis:")
    
    # Symmetric tree
    sym_arr = [1, 2, 2, 3, 4, 4, 3]
    sym_tree = analyzer.build_tree_from_array(sym_arr)
    
    print("Symmetric test tree:")
    analyzer.print_tree_structure(sym_tree)
    
    is_sym1 = analyzer.is_symmetric_recursive(sym_tree)
    is_sym2 = analyzer.is_symmetric_iterative(sym_tree)
    is_sym3 = analyzer.is_symmetric_level_check(sym_tree)
    
    print(f"Is symmetric (recursive): {is_sym1}")
    print(f"Is symmetric (iterative): {is_sym2}")
    print(f"Is symmetric (level check): {is_sym3}")
    
    # Non-symmetric tree
    non_sym_arr = [1, 2, 2, None, 3, None, 3]
    non_sym_tree = analyzer.build_tree_from_array(non_sym_arr)
    
    print("Non-symmetric test tree:")
    analyzer.print_tree_structure(non_sym_tree)
    print(f"Is symmetric: {analyzer.is_symmetric_recursive(non_sym_tree)}")
    print()
    
    # Example 4: Subtree Analysis
    print("4. Subtree Analysis:")
    
    main_arr = [3, 4, 5, 1, 2]
    sub_arr = [4, 1, 2]
    
    main_tree = analyzer.build_tree_from_array(main_arr)
    sub_tree = analyzer.build_tree_from_array(sub_arr)
    
    print("Main tree:")
    analyzer.print_tree_structure(main_tree)
    print("Potential subtree:")
    analyzer.print_tree_structure(sub_tree)
    
    is_sub1 = analyzer.is_subtree_naive(main_tree, sub_tree)
    is_sub2 = analyzer.is_subtree_serialization(main_tree, sub_tree)
    is_sub3 = analyzer.is_subtree_hash(main_tree, sub_tree)
    
    print(f"Is subtree (naive): {is_sub1}")
    print(f"Is subtree (serialization): {is_sub2}")
    print(f"Is subtree (hash): {is_sub3}")
    print()
    
    # Example 5: Tree Isomorphism
    print("5. Tree Isomorphism Analysis:")
    
    # Two isomorphic trees
    iso1_arr = [1, 2, 3, 4, 5]
    iso2_arr = [1, 3, 2, None, None, 4, 5]
    
    iso_tree1 = analyzer.build_tree_from_array(iso1_arr)
    iso_tree2 = analyzer.build_tree_from_array(iso2_arr)
    
    print("Tree 1:")
    analyzer.print_tree_structure(iso_tree1)
    print("Tree 2:")
    analyzer.print_tree_structure(iso_tree2)
    
    is_iso1 = analyzer.is_isomorphic_recursive(iso_tree1, iso_tree2)
    is_iso2 = analyzer.is_isomorphic_canonical(iso_tree1, iso_tree2)
    
    print(f"Are isomorphic (recursive): {is_iso1}")
    print(f"Are isomorphic (canonical): {is_iso2}")
    print()
    
    # Example 6: Comprehensive Tree Statistics
    print("6. Comprehensive Tree Statistics:")
    
    complex_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    complex_tree = analyzer.build_tree_from_array(complex_arr)
    
    print("Complex tree:")
    analyzer.print_tree_structure(complex_tree)
    
    stats = analyzer.get_tree_statistics(complex_tree)
    print("Tree Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Example 7: Tree Property Tests
    print("7. Tree Property Tests:")
    
    # Perfect tree
    perfect_arr = [1, 2, 3, 4, 5, 6, 7]
    perfect_tree = analyzer.build_tree_from_array(perfect_arr)
    
    print("Perfect tree test:")
    analyzer.print_tree_structure(perfect_tree)
    print(f"Is complete: {analyzer.is_complete_tree(perfect_tree)}")
    print(f"Is perfect: {analyzer.is_perfect_tree(perfect_tree)}")
    
    # Complete but not perfect tree
    complete_arr = [1, 2, 3, 4, 5, 6]
    complete_tree = analyzer.build_tree_from_array(complete_arr)
    
    print("Complete tree test:")
    analyzer.print_tree_structure(complete_tree)
    print(f"Is complete: {analyzer.is_complete_tree(complete_tree)}")
    print(f"Is perfect: {analyzer.is_perfect_tree(complete_tree)}")
    
    print("\n=== Demo Complete ===") 