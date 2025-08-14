"""
Balanced Trees - AVL Trees and Red-Black Trees
==============================================

Topics: Self-balancing binary search trees, rotations, balance maintenance
Companies: Google, Amazon, Microsoft, Facebook, Apple, Netflix
Difficulty: Hard to Expert
Time Complexity: O(log n) for all operations
Space Complexity: O(n) for tree storage, O(log n) for recursion
"""

from typing import List, Optional, Dict, Any, Tuple, Union
from collections import deque
import math

class BalancedTrees:
    
    def __init__(self):
        """Initialize with balanced tree tracking"""
        self.operation_count = 0
        self.rotation_stats = {}
    
    # ==========================================
    # 1. AVL TREE FUNDAMENTALS
    # ==========================================
    
    def explain_avl_concept(self) -> None:
        """
        Explain AVL tree concepts and properties
        
        AVL trees maintain balance through height tracking and rotations
        """
        print("=== AVL TREE FUNDAMENTALS ===")
        print("AVL Tree: Self-balancing binary search tree with strict height balance")
        print()
        print("KEY PROPERTIES:")
        print("• Height difference between left and right subtrees ≤ 1")
        print("• All BST properties maintained")
        print("• Automatic rebalancing through rotations")
        print("• Guaranteed O(log n) operations")
        print("• Named after Adelson-Velsky and Landis (1962)")
        print()
        print("BALANCE FACTOR:")
        print("• BF(node) = height(left_subtree) - height(right_subtree)")
        print("• Valid balance factors: -1, 0, +1")
        print("• BF = +1: Left subtree is taller")
        print("• BF =  0: Both subtrees have equal height")
        print("• BF = -1: Right subtree is taller")
        print()
        print("ROTATION TYPES:")
        print("• Single Right Rotation (LL case)")
        print("• Single Left Rotation (RR case)")
        print("• Left-Right Rotation (LR case)")
        print("• Right-Left Rotation (RL case)")
        print()
        print("WHEN TO USE AVL TREES:")
        print("• Frequent lookups required")
        print("• Need guaranteed O(log n) operations")
        print("• Height balance is critical")
        print("• Read-heavy applications")
        print()
        print("ADVANTAGES:")
        print("• Strictly balanced (height difference ≤ 1)")
        print("• Predictable performance")
        print("• Excellent for search-intensive applications")
        print()
        print("DISADVANTAGES:")
        print("• More rotations than Red-Black trees")
        print("• Higher insertion/deletion overhead")
        print("• More complex implementation")
    
    def demonstrate_avl_operations(self) -> None:
        """
        Demonstrate AVL tree operations with detailed explanation
        """
        print("=== AVL TREE OPERATIONS DEMONSTRATION ===")
        print("Demonstrating insertion with automatic balancing")
        print()
        
        avl = AVLTree()
        
        # Test sequence that triggers all rotation types
        values = [10, 20, 30, 40, 50, 25]
        
        print("Inserting values:", values)
        print()
        
        for value in values:
            print(f"Inserting {value}:")
            avl.insert(value)
            print()
        
        print("Final AVL tree structure:")
        avl.display_tree()
        
        print("\nAVL tree properties verification:")
        print(f"  Height: {avl.get_height()}")
        print(f"  Is balanced: {avl.is_balanced()}")
        print(f"  Total rotations performed: {avl.rotation_count}")
        
        # Test deletion
        print(f"\nDeleting values:")
        delete_values = [20, 30]
        for value in delete_values:
            print(f"Deleting {value}:")
            avl.delete(value)
            print()
        
        print("Tree after deletions:")
        avl.display_tree()


class AVLNode:
    """
    AVL Tree Node with height information
    
    Stores data, left/right children, and height for balance calculations
    """
    
    def __init__(self, data: int):
        self.data = data
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height = 1  # Height of the node
    
    def __str__(self):
        return f"AVL({self.data}, h:{self.height})"


class AVLTree:
    """
    Complete AVL Tree implementation with all operations
    
    Features:
    - Automatic balancing through rotations
    - Height tracking for balance factor calculation
    - Complete CRUD operations
    - Tree visualization and validation
    """
    
    def __init__(self):
        self.root: Optional[AVLNode] = None
        self.rotation_count = 0
        self.operation_count = 0
    
    def _get_height(self, node: Optional[AVLNode]) -> int:
        """Get height of a node (0 for None)"""
        return node.height if node else 0
    
    def _get_balance_factor(self, node: Optional[AVLNode]) -> int:
        """Calculate balance factor: height(left) - height(right)"""
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _update_height(self, node: AVLNode) -> None:
        """Update height based on children's heights"""
        left_height = self._get_height(node.left)
        right_height = self._get_height(node.right)
        node.height = 1 + max(left_height, right_height)
    
    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """
        Right rotation (LL case)
        
        Before:      After:
           y            x
          / \          / \
         x   T3       T1  y
        / \              / \
       T1  T2           T2  T3
        """
        print(f"    Performing RIGHT rotation at node {y.data}")
        
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        self._update_height(y)
        self._update_height(x)
        
        self.rotation_count += 1
        
        print(f"    Rotation complete: {x.data} is new root of subtree")
        return x
    
    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """
        Left rotation (RR case)
        
        Before:      After:
           x            y
          / \          / \
         T1  y        x   T3
            / \      / \
           T2  T3   T1  T2
        """
        print(f"    Performing LEFT rotation at node {x.data}")
        
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        self._update_height(x)
        self._update_height(y)
        
        self.rotation_count += 1
        
        print(f"    Rotation complete: {y.data} is new root of subtree")
        return y
    
    def insert(self, data: int) -> None:
        """Insert data into AVL tree with automatic balancing"""
        self.root = self._insert_recursive(self.root, data)
        self.operation_count += 1
    
    def _insert_recursive(self, node: Optional[AVLNode], data: int) -> AVLNode:
        """Recursive insertion with balancing"""
        # Standard BST insertion
        if not node:
            print(f"  Creating new node: {data}")
            return AVLNode(data)
        
        if data < node.data:
            print(f"  Going left from {node.data}")
            node.left = self._insert_recursive(node.left, data)
        elif data > node.data:
            print(f"  Going right from {node.data}")
            node.right = self._insert_recursive(node.right, data)
        else:
            print(f"  Duplicate value {data}, not inserting")
            return node
        
        # Update height of current node
        self._update_height(node)
        
        # Get balance factor
        balance = self._get_balance_factor(node)
        
        print(f"  Node {node.data}: height={node.height}, balance={balance}")
        
        # Check if rebalancing is needed
        if abs(balance) > 1:
            print(f"  ⚠️  Imbalance detected at node {node.data} (balance={balance})")
            return self._rebalance(node, balance, data)
        
        return node
    
    def _rebalance(self, node: AVLNode, balance: int, inserted_data: int) -> AVLNode:
        """Rebalance the tree using appropriate rotations"""
        
        # Left Heavy (balance > 1)
        if balance > 1:
            # Left-Left case
            if inserted_data < node.left.data:
                print(f"  Detected LL case")
                return self._rotate_right(node)
            
            # Left-Right case
            if inserted_data > node.left.data:
                print(f"  Detected LR case")
                print(f"  First: left rotation at {node.left.data}")
                node.left = self._rotate_left(node.left)
                print(f"  Second: right rotation at {node.data}")
                return self._rotate_right(node)
        
        # Right Heavy (balance < -1)
        if balance < -1:
            # Right-Right case
            if inserted_data > node.right.data:
                print(f"  Detected RR case")
                return self._rotate_left(node)
            
            # Right-Left case
            if inserted_data < node.right.data:
                print(f"  Detected RL case")
                print(f"  First: right rotation at {node.right.data}")
                node.right = self._rotate_right(node.right)
                print(f"  Second: left rotation at {node.data}")
                return self._rotate_left(node)
        
        return node
    
    def delete(self, data: int) -> None:
        """Delete data from AVL tree with rebalancing"""
        self.root = self._delete_recursive(self.root, data)
        self.operation_count += 1
    
    def _delete_recursive(self, node: Optional[AVLNode], data: int) -> Optional[AVLNode]:
        """Recursive deletion with balancing"""
        if not node:
            print(f"  Value {data} not found")
            return None
        
        # Standard BST deletion
        if data < node.data:
            node.left = self._delete_recursive(node.left, data)
        elif data > node.data:
            node.right = self._delete_recursive(node.right, data)
        else:
            print(f"  Found node to delete: {data}")
            
            # Node with only one child or no child
            if not node.left:
                print(f"  Node has only right child (or no children)")
                return node.right
            elif not node.right:
                print(f"  Node has only left child")
                return node.left
            
            # Node with two children
            print(f"  Node has both children - finding inorder successor")
            successor = self._find_min(node.right)
            print(f"  Inorder successor: {successor.data}")
            
            # Copy successor's data to this node
            node.data = successor.data
            
            # Delete the successor
            node.right = self._delete_recursive(node.right, successor.data)
        
        # Update height
        self._update_height(node)
        
        # Get balance factor
        balance = self._get_balance_factor(node)
        
        print(f"  After deletion - Node {node.data}: height={node.height}, balance={balance}")
        
        # Rebalance if necessary
        if abs(balance) > 1:
            print(f"  ⚠️  Imbalance detected after deletion at node {node.data}")
            return self._rebalance_after_deletion(node, balance)
        
        return node
    
    def _rebalance_after_deletion(self, node: AVLNode, balance: int) -> AVLNode:
        """Rebalance after deletion operation"""
        
        # Left Heavy
        if balance > 1:
            left_balance = self._get_balance_factor(node.left)
            
            # Left-Left case
            if left_balance >= 0:
                print(f"  Post-deletion LL case")
                return self._rotate_right(node)
            
            # Left-Right case
            else:
                print(f"  Post-deletion LR case")
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        
        # Right Heavy
        if balance < -1:
            right_balance = self._get_balance_factor(node.right)
            
            # Right-Right case
            if right_balance <= 0:
                print(f"  Post-deletion RR case")
                return self._rotate_left(node)
            
            # Right-Left case
            else:
                print(f"  Post-deletion RL case")
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        
        return node
    
    def _find_min(self, node: AVLNode) -> AVLNode:
        """Find minimum value node in subtree"""
        while node.left:
            node = node.left
        return node
    
    def search(self, data: int) -> bool:
        """Search for data in AVL tree"""
        return self._search_recursive(self.root, data)
    
    def _search_recursive(self, node: Optional[AVLNode], data: int) -> bool:
        """Recursive search"""
        if not node:
            return False
        
        if data == node.data:
            return True
        elif data < node.data:
            return self._search_recursive(node.left, data)
        else:
            return self._search_recursive(node.right, data)
    
    def get_height(self) -> int:
        """Get height of the tree"""
        return self._get_height(self.root)
    
    def is_balanced(self) -> bool:
        """Check if tree is balanced (all balance factors are -1, 0, or 1)"""
        return self._is_balanced_recursive(self.root)
    
    def _is_balanced_recursive(self, node: Optional[AVLNode]) -> bool:
        """Recursive balance check"""
        if not node:
            return True
        
        balance = self._get_balance_factor(node)
        if abs(balance) > 1:
            return False
        
        return (self._is_balanced_recursive(node.left) and 
                self._is_balanced_recursive(node.right))
    
    def display_tree(self) -> None:
        """Display tree structure with heights and balance factors"""
        if not self.root:
            print("  (empty tree)")
            return
        
        print("  Tree structure (data[height:balance]):")
        self._display_recursive(self.root, "", True)
    
    def _display_recursive(self, node: Optional[AVLNode], prefix: str, is_last: bool) -> None:
        """Recursive tree display"""
        if node:
            print(f"  {prefix}{'└── ' if is_last else '├── '}{node.data}[h:{node.height}, b:{self._get_balance_factor(node)}]")
            
            if node.left or node.right:
                if node.right:
                    self._display_recursive(node.right, prefix + ("    " if is_last else "│   "), not bool(node.left))
                if node.left:
                    self._display_recursive(node.left, prefix + ("    " if is_last else "│   "), True)


# ==========================================
# 2. RED-BLACK TREE FUNDAMENTALS
# ==========================================

class RedBlackTrees:
    """Red-Black Trees implementation and explanation"""
    
    def explain_red_black_concept(self) -> None:
        """
        Explain Red-Black tree concepts and properties
        """
        print("=== RED-BLACK TREE FUNDAMENTALS ===")
        print("Red-Black Tree: Self-balancing BST with color-based balance rules")
        print()
        print("RED-BLACK PROPERTIES:")
        print("1. Every node is either RED or BLACK")
        print("2. Root is always BLACK")
        print("3. All leaves (NIL nodes) are BLACK")
        print("4. RED nodes cannot have RED children (no consecutive reds)")
        print("5. Every path from node to descendant leaves has same number of BLACK nodes")
        print()
        print("BALANCE GUARANTEE:")
        print("• Height is at most 2 * log₂(n + 1)")
        print("• Longest path ≤ 2 × shortest path")
        print("• Less strict than AVL, but still guarantees O(log n)")
        print()
        print("INSERTION CASES:")
        print("• Case 1: Node is root → color it BLACK")
        print("• Case 2: Parent is BLACK → no changes needed")
        print("• Case 3: Parent and uncle are RED → recolor")
        print("• Case 4: Parent is RED, uncle is BLACK → rotate")
        print()
        print("ADVANTAGES OVER AVL:")
        print("• Fewer rotations during insertion/deletion")
        print("• Better for write-heavy applications")
        print("• Used in many standard libraries (C++ STL, Java TreeMap)")
        print()
        print("REAL-WORLD USAGE:")
        print("• C++ std::map and std::set")
        print("• Java TreeMap and TreeSet")
        print("• Linux kernel's Completely Fair Scheduler")
        print("• Database indexing systems")
    
    def demonstrate_red_black_operations(self) -> None:
        """Demonstrate Red-Black tree operations"""
        print("=== RED-BLACK TREE OPERATIONS DEMONSTRATION ===")
        
        rb_tree = RedBlackTree()
        
        values = [7, 3, 18, 10, 22, 8, 11, 26]
        
        print(f"Inserting values: {values}")
        print()
        
        for value in values:
            print(f"Inserting {value}:")
            rb_tree.insert(value)
            print(f"  Tree is valid: {rb_tree.is_valid()}")
            print()
        
        print("Final Red-Black tree structure:")
        rb_tree.display_tree()
        
        print(f"\nTree properties:")
        print(f"  Height: {rb_tree.get_height()}")
        print(f"  Black height: {rb_tree.get_black_height()}")
        print(f"  Is valid Red-Black tree: {rb_tree.is_valid()}")


class Color:
    """Color enumeration for Red-Black tree nodes"""
    RED = "RED"
    BLACK = "BLACK"


class RBNode:
    """Red-Black Tree Node"""
    
    def __init__(self, data: int, color: str = Color.RED):
        self.data = data
        self.color = color
        self.left: Optional['RBNode'] = None
        self.right: Optional['RBNode'] = None
        self.parent: Optional['RBNode'] = None
    
    def __str__(self):
        return f"RB({self.data}, {self.color})"


class RedBlackTree:
    """
    Red-Black Tree implementation with insertion and validation
    
    Simplified implementation focusing on core concepts
    """
    
    def __init__(self):
        self.NIL = RBNode(0, Color.BLACK)  # Sentinel NIL node
        self.root = self.NIL
    
    def insert(self, data: int) -> None:
        """Insert data into Red-Black tree"""
        new_node = RBNode(data, Color.RED)
        new_node.left = self.NIL
        new_node.right = self.NIL
        
        # Standard BST insertion
        parent = None
        current = self.root
        
        while current != self.NIL:
            parent = current
            if data < current.data:
                current = current.left
            elif data > current.data:
                current = current.right
            else:
                print(f"  Duplicate value {data}, not inserting")
                return
        
        new_node.parent = parent
        
        if parent is None:
            self.root = new_node
        elif data < parent.data:
            parent.left = new_node
        else:
            parent.right = new_node
        
        print(f"  Inserted {data} as RED node")
        
        # Fix Red-Black tree properties
        self._fix_insert(new_node)
    
    def _fix_insert(self, node: RBNode) -> None:
        """Fix Red-Black tree properties after insertion"""
        print(f"  Fixing Red-Black properties for node {node.data}")
        
        while node.parent and node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                
                if uncle.color == Color.RED:
                    # Case 3: Parent and uncle are RED
                    print(f"    Case 3: Recoloring parent, uncle, and grandparent")
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 4a: Left rotation needed
                        print(f"    Case 4a: Left rotation at {node.parent.data}")
                        node = node.parent
                        self._rotate_left(node)
                    
                    # Case 4b: Right rotation and recoloring
                    print(f"    Case 4b: Right rotation and recoloring")
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._rotate_right(node.parent.parent)
            else:
                # Symmetric cases for right subtree
                uncle = node.parent.parent.left
                
                if uncle.color == Color.RED:
                    print(f"    Case 3 (symmetric): Recoloring")
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        print(f"    Case 4a (symmetric): Right rotation")
                        node = node.parent
                        self._rotate_right(node)
                    
                    print(f"    Case 4b (symmetric): Left rotation and recoloring")
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._rotate_left(node.parent.parent)
        
        # Root must always be BLACK
        self.root.color = Color.BLACK
        print(f"  Root colored BLACK")
    
    def _rotate_left(self, x: RBNode) -> None:
        """Left rotation"""
        y = x.right
        x.right = y.left
        
        if y.left != self.NIL:
            y.left.parent = x
        
        y.parent = x.parent
        
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x
        x.parent = y
    
    def _rotate_right(self, y: RBNode) -> None:
        """Right rotation"""
        x = y.left
        y.left = x.right
        
        if x.right != self.NIL:
            x.right.parent = y
        
        x.parent = y.parent
        
        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        
        x.right = y
        y.parent = x
    
    def is_valid(self) -> bool:
        """Validate Red-Black tree properties"""
        return (self._check_property_1() and
                self._check_property_2() and
                self._check_property_4() and
                self._check_property_5())
    
    def _check_property_1(self) -> bool:
        """Check: Every node is RED or BLACK"""
        def check_node(node):
            if node == self.NIL:
                return True
            return (node.color in [Color.RED, Color.BLACK] and
                    check_node(node.left) and
                    check_node(node.right))
        
        return check_node(self.root)
    
    def _check_property_2(self) -> bool:
        """Check: Root is BLACK"""
        return self.root == self.NIL or self.root.color == Color.BLACK
    
    def _check_property_4(self) -> bool:
        """Check: No consecutive RED nodes"""
        def check_node(node):
            if node == self.NIL:
                return True
            
            if node.color == Color.RED:
                if (node.left.color == Color.RED or 
                    node.right.color == Color.RED):
                    return False
            
            return check_node(node.left) and check_node(node.right)
        
        return check_node(self.root)
    
    def _check_property_5(self) -> bool:
        """Check: Same number of BLACK nodes on all paths"""
        def get_black_height(node):
            if node == self.NIL:
                return 0
            
            left_height = get_black_height(node.left)
            right_height = get_black_height(node.right)
            
            if left_height != right_height or left_height == -1:
                return -1
            
            return left_height + (1 if node.color == Color.BLACK else 0)
        
        return get_black_height(self.root) != -1
    
    def get_height(self) -> int:
        """Get height of the tree"""
        def height_recursive(node):
            if node == self.NIL:
                return 0
            return 1 + max(height_recursive(node.left), height_recursive(node.right))
        
        return height_recursive(self.root)
    
    def get_black_height(self) -> int:
        """Get black height of the tree"""
        def black_height_recursive(node):
            if node == self.NIL:
                return 0
            
            left_height = black_height_recursive(node.left)
            return left_height + (1 if node.color == Color.BLACK else 0)
        
        return black_height_recursive(self.root)
    
    def display_tree(self) -> None:
        """Display tree structure with colors"""
        if self.root == self.NIL:
            print("  (empty tree)")
            return
        
        print("  Tree structure (data[color]):")
        self._display_recursive(self.root, "", True)
    
    def _display_recursive(self, node: RBNode, prefix: str, is_last: bool) -> None:
        """Recursive tree display"""
        if node != self.NIL:
            color_symbol = "🔴" if node.color == Color.RED else "⚫"
            print(f"  {prefix}{'└── ' if is_last else '├── '}{node.data}{color_symbol}")
            
            if node.left != self.NIL or node.right != self.NIL:
                if node.right != self.NIL:
                    self._display_recursive(node.right, prefix + ("    " if is_last else "│   "), node.left == self.NIL)
                if node.left != self.NIL:
                    self._display_recursive(node.left, prefix + ("    " if is_last else "│   "), True)


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_balanced_trees():
    """Demonstrate all balanced tree concepts"""
    print("=== BALANCED TREES DEMONSTRATION ===\n")
    
    balanced_trees = BalancedTrees()
    
    # 1. AVL Tree concepts and operations
    balanced_trees.explain_avl_concept()
    print("\n" + "="*60 + "\n")
    
    balanced_trees.demonstrate_avl_operations()
    print("\n" + "="*60 + "\n")
    
    # 2. Red-Black Tree concepts and operations
    rb_trees = RedBlackTrees()
    rb_trees.explain_red_black_concept()
    print("\n" + "="*60 + "\n")
    
    rb_trees.demonstrate_red_black_operations()
    print("\n" + "="*60 + "\n")
    
    # 3. Performance comparison
    print("=== AVL vs RED-BLACK COMPARISON ===")
    
    print("Performance Characteristics:")
    print()
    print("┌─────────────────┬─────────────┬─────────────────┐")
    print("│ Operation       │ AVL Tree    │ Red-Black Tree  │")
    print("├─────────────────┼─────────────┼─────────────────┤")
    print("│ Search          │ O(log n)    │ O(log n)        │")
    print("│ Insert          │ O(log n)    │ O(log n)        │")
    print("│ Delete          │ O(log n)    │ O(log n)        │")
    print("│ Height bound    │ 1.44 log n  │ 2 log n         │")
    print("│ Rotations/insert│ ≤ 2         │ ≤ 2             │")
    print("│ Rotations/delete│ O(log n)    │ ≤ 3             │")
    print("└─────────────────┴─────────────┴─────────────────┘")
    print()
    
    print("When to use each:")
    print()
    print("AVL Trees:")
    print("  ✓ Search-intensive applications")
    print("  ✓ Need strict height balance")
    print("  ✓ Fewer insertions/deletions")
    print("  ✓ Predictable performance required")
    print()
    print("Red-Black Trees:")
    print("  ✓ Write-heavy applications")
    print("  ✓ Standard library implementations")
    print("  ✓ General-purpose balanced BST")
    print("  ✓ Better insertion/deletion performance")


if __name__ == "__main__":
    demonstrate_balanced_trees()
    
    print("\n=== BALANCED TREES MASTERY GUIDE ===")
    
    print("\n🎯 KEY CONCEPTS TO MASTER:")
    print("• Height balancing vs color-based balancing")
    print("• Rotation mechanics and when to apply them")
    print("• Balance factor calculation and maintenance")
    print("• Red-Black properties and violation handling")
    print("• Performance trade-offs between different approaches")
    
    print("\n📊 COMPLEXITY GUARANTEES:")
    print("• All operations: O(log n) for both AVL and Red-Black")
    print("• AVL height: ≤ 1.44 log₂(n + 2) - 0.328")
    print("• Red-Black height: ≤ 2 log₂(n + 1)")
    print("• AVL rotations: at most 2 per insertion, O(log n) per deletion")
    print("• Red-Black rotations: at most 2 per insertion, at most 3 per deletion")
    
    print("\n⚡ IMPLEMENTATION STRATEGIES:")
    print("• Choose AVL for search-heavy workloads")
    print("• Choose Red-Black for balanced read/write workloads")
    print("• Implement proper rotation mechanics")
    print("• Maintain invariants after every modification")
    print("• Add comprehensive validation for debugging")
    
    print("\n🔧 ADVANCED TECHNIQUES:")
    print("• Lazy balancing for batch operations")
    print("• Thread-safe balanced tree implementations")
    print("• Persistent balanced trees for functional programming")
    print("• Range query optimizations")
    print("• Memory pool allocation for better performance")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Understand basic BST operations and limitations")
    print("2. Master rotation mechanics and balance maintenance")
    print("3. Implement AVL trees with full rebalancing")
    print("4. Learn Red-Black properties and color-based balancing")
    print("5. Study advanced variations and optimizations")
    
    print("\n🏆 REAL-WORLD APPLICATIONS:")
    print("• Database indexing (B+ trees, Red-Black trees)")
    print("• Standard library containers (std::map, TreeMap)")
    print("• Operating system schedulers")
    print("• Compiler symbol tables")
    print("• Network routing algorithms")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Practice drawing rotations by hand")
    print("• Implement validation functions for debugging")
    print("• Understand the invariants deeply")
    print("• Study real-world implementations")
    print("• Compare performance characteristics empirically")
