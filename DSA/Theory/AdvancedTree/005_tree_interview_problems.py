"""
Tree Interview Problems - Advanced Patterns
==========================================

Topics: LeetCode hard problems, competitive programming, interview strategies
Companies: FAANG+ companies, competitive programming contests
Difficulty: Hard to Expert
Time Complexity: O(log n) to O(n²) depending on problem
Space Complexity: O(log n) to O(n) for recursion and storage
"""

from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict, deque
import heapq
import math

class TreeInterviewProblems:
    
    def __init__(self):
        """Initialize with problem tracking"""
        self.problems_solved = 0
        self.patterns_learned = set()
    
    # ==========================================
    # 1. ADVANCED TREE CONSTRUCTION PROBLEMS
    # ==========================================
    
    def explain_construction_patterns(self) -> None:
        """
        Explain advanced tree construction patterns
        """
        print("=== ADVANCED TREE CONSTRUCTION PATTERNS ===")
        print("Complex tree building from various inputs")
        print()
        print("CONSTRUCTION FROM TRAVERSALS:")
        print("• Inorder + Preorder → Unique binary tree")
        print("• Inorder + Postorder → Unique binary tree")
        print("• Preorder + Postorder → Not unique (need additional info)")
        print("• Level order + Inorder → Unique binary tree")
        print()
        print("CONSTRUCTION FROM ARRAYS:")
        print("• Array to BST: Choose middle as root recursively")
        print("• Array to balanced tree: Minimize height")
        print("• Heap array to tree: Use heap property")
        print("• Serialized format to tree: Parse structure")
        print()
        print("CONSTRUCTION FROM SPECIAL CONDITIONS:")
        print("• Maximum binary tree from array")
        print("• Tree from parent array representation")
        print("• BST from sorted list (minimize height)")
        print("• Complete binary tree from array")
        print()
        print("KEY STRATEGIES:")
        print("• Use recursion with proper base cases")
        print("• Identify root selection criteria")
        print("• Partition input correctly for left/right subtrees")
        print("• Handle edge cases (empty input, single node)")
    
    def demonstrate_construction_problems(self) -> None:
        """
        Demonstrate complex tree construction problems
        """
        print("=== TREE CONSTRUCTION DEMONSTRATIONS ===")
        
        constructor = TreeConstructor()
        
        # Problem 1: Construct from inorder and preorder
        print("1. CONSTRUCT BINARY TREE FROM INORDER AND PREORDER")
        inorder = [9, 3, 15, 20, 7]
        preorder = [3, 9, 20, 15, 7]
        
        print(f"   Inorder: {inorder}")
        print(f"   Preorder: {preorder}")
        
        tree1 = constructor.build_tree_from_traversals(preorder, inorder)
        print("   Constructed tree (level order):")
        constructor.display_level_order(tree1)
        print()
        
        # Problem 2: Maximum binary tree
        print("2. MAXIMUM BINARY TREE")
        arr = [3, 2, 1, 6, 0, 5]
        print(f"   Array: {arr}")
        print("   Rule: Root is max element, recursively build left/right")
        
        tree2 = constructor.construct_maximum_binary_tree(arr)
        print("   Constructed tree (level order):")
        constructor.display_level_order(tree2)
        print()
        
        # Problem 3: BST from sorted array
        print("3. BALANCED BST FROM SORTED ARRAY")
        sorted_arr = [-10, -3, 0, 5, 9]
        print(f"   Sorted array: {sorted_arr}")
        
        tree3 = constructor.sorted_array_to_bst(sorted_arr)
        print("   Constructed balanced BST (level order):")
        constructor.display_level_order(tree3)
        print("   Verifying balance:")
        print(f"     Height: {constructor.get_height(tree3)}")
        print(f"     Is balanced: {constructor.is_balanced(tree3)}")


class TreeNode:
    """Standard binary tree node"""
    
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __str__(self):
        return str(self.val)


class TreeConstructor:
    """
    Advanced tree construction algorithms
    """
    
    def build_tree_from_traversals(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        """
        LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal
        
        Time: O(n), Space: O(n)
        """
        if not preorder or not inorder:
            return None
        
        # Create mapping for quick lookup of inorder positions
        inorder_map = {val: i for i, val in enumerate(inorder)}
        self.preorder_idx = 0
        
        def build_recursive(in_start, in_end):
            if in_start > in_end:
                return None
            
            # Root is always the current element in preorder
            root_val = preorder[self.preorder_idx]
            self.preorder_idx += 1
            
            root = TreeNode(root_val)
            
            # Find root position in inorder
            root_pos = inorder_map[root_val]
            
            # Build left subtree first (preorder nature)
            root.left = build_recursive(in_start, root_pos - 1)
            root.right = build_recursive(root_pos + 1, in_end)
            
            return root
        
        return build_recursive(0, len(inorder) - 1)
    
    def construct_maximum_binary_tree(self, nums: List[int]) -> Optional[TreeNode]:
        """
        LeetCode 654: Maximum Binary Tree
        
        Time: O(n²) worst case, O(n log n) average
        Space: O(n) for recursion
        """
        if not nums:
            return None
        
        # Find maximum element
        max_val = max(nums)
        max_idx = nums.index(max_val)
        
        # Create root with maximum value
        root = TreeNode(max_val)
        
        # Recursively build left and right subtrees
        root.left = self.construct_maximum_binary_tree(nums[:max_idx])
        root.right = self.construct_maximum_binary_tree(nums[max_idx + 1:])
        
        return root
    
    def sorted_array_to_bst(self, nums: List[int]) -> Optional[TreeNode]:
        """
        LeetCode 108: Convert Sorted Array to Binary Search Tree
        
        Time: O(n), Space: O(log n) for recursion
        """
        def build_balanced(left, right):
            if left > right:
                return None
            
            # Choose middle as root to maintain balance
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            
            # Recursively build left and right subtrees
            root.left = build_balanced(left, mid - 1)
            root.right = build_balanced(mid + 1, right)
            
            return root
        
        return build_balanced(0, len(nums) - 1)
    
    def display_level_order(self, root: Optional[TreeNode]) -> None:
        """Display tree in level order"""
        if not root:
            print("     (empty tree)")
            return
        
        queue = deque([root])
        level = 0
        
        while queue:
            level_size = len(queue)
            level_values = []
            
            for _ in range(level_size):
                node = queue.popleft()
                if node:
                    level_values.append(str(node.val))
                    queue.append(node.left)
                    queue.append(node.right)
                else:
                    level_values.append("null")
            
            # Remove trailing nulls
            while level_values and level_values[-1] == "null":
                level_values.pop()
            
            if level_values:
                print(f"     Level {level}: {level_values}")
                level += 1
            
            # Stop if no more non-null nodes
            if all(node is None for node in queue):
                break
    
    def get_height(self, root: Optional[TreeNode]) -> int:
        """Get height of tree"""
        if not root:
            return 0
        return 1 + max(self.get_height(root.left), self.get_height(root.right))
    
    def is_balanced(self, root: Optional[TreeNode]) -> bool:
        """Check if tree is height-balanced"""
        def check_balance(node):
            if not node:
                return 0, True
            
            left_height, left_balanced = check_balance(node.left)
            right_height, right_balanced = check_balance(node.right)
            
            balanced = (left_balanced and right_balanced and 
                       abs(left_height - right_height) <= 1)
            height = 1 + max(left_height, right_height)
            
            return height, balanced
        
        _, balanced = check_balance(root)
        return balanced


# ==========================================
# 2. TREE MODIFICATION AND TRANSFORMATION
# ==========================================

class TreeTransformation:
    """
    Advanced tree modification and transformation problems
    """
    
    def explain_transformation_patterns(self) -> None:
        """Explain tree transformation patterns"""
        print("=== TREE TRANSFORMATION PATTERNS ===")
        print("Modifying tree structure while maintaining properties")
        print()
        print("COMMON TRANSFORMATION TYPES:")
        print("• Tree flattening: Convert to linked list structure")
        print("• Tree inversion: Mirror/flip tree horizontally")
        print("• Tree pruning: Remove nodes based on conditions")
        print("• Tree grafting: Merge or attach subtrees")
        print("• In-place modifications: Change structure without extra space")
        print()
        print("FLATTENING STRATEGIES:")
        print("• Preorder flattening: Root → Left → Right")
        print("• Inorder flattening: Left → Root → Right")
        print("• Postorder flattening: Left → Right → Root")
        print("• Right-skewed: All nodes have only right children")
        print()
        print("KEY TECHNIQUES:")
        print("• Use Morris traversal for O(1) space")
        print("• Maintain pointers during recursive modification")
        print("• Consider both recursive and iterative approaches")
        print("• Handle edge cases carefully")
    
    def demonstrate_transformation_problems(self) -> None:
        """Demonstrate tree transformation problems"""
        print("=== TREE TRANSFORMATION DEMONSTRATIONS ===")
        
        transformer = TreeTransformer()
        
        # Create sample tree
        root = TreeNode(1)
        root.left = TreeNode(2)
        root.right = TreeNode(5)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(4)
        root.right.right = TreeNode(6)
        
        print("Original tree:")
        print("     1")
        print("    / \\")
        print("   2   5")
        print("  / \\   \\")
        print(" 3   4   6")
        print()
        
        # Problem 1: Flatten binary tree to linked list
        print("1. FLATTEN BINARY TREE TO LINKED LIST")
        print("   Converting to right-skewed tree (preorder sequence)")
        
        # Make a copy for demonstration
        root_copy1 = transformer.copy_tree(root)
        transformer.flatten_to_linked_list(root_copy1)
        
        print("   Flattened tree (preorder sequence):")
        transformer.display_flattened(root_copy1)
        print()
        
        # Problem 2: Invert binary tree
        print("2. INVERT BINARY TREE")
        print("   Mirroring tree horizontally")
        
        root_copy2 = transformer.copy_tree(root)
        inverted = transformer.invert_tree(root_copy2)
        
        print("   Inverted tree structure:")
        transformer.display_tree_structure(inverted)
        print()
        
        # Problem 3: Prune tree based on sum
        print("3. BINARY TREE PRUNING")
        print("   Remove subtrees where all nodes have value < 4")
        
        root_copy3 = transformer.copy_tree(root)
        pruned = transformer.prune_tree(root_copy3, 4)
        
        print("   Pruned tree:")
        transformer.display_tree_structure(pruned)


class TreeTransformer:
    """
    Tree transformation algorithms implementation
    """
    
    def flatten_to_linked_list(self, root: Optional[TreeNode]) -> None:
        """
        LeetCode 114: Flatten Binary Tree to Linked List
        
        Time: O(n), Space: O(1) using Morris-like approach
        """
        if not root:
            return
        
        current = root
        
        while current:
            if current.left:
                # Find rightmost node in left subtree
                predecessor = current.left
                while predecessor.right:
                    predecessor = predecessor.right
                
                # Connect rightmost to current's right
                predecessor.right = current.right
                
                # Move left subtree to right
                current.right = current.left
                current.left = None
            
            # Move to next node
            current = current.right
    
    def invert_tree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        LeetCode 226: Invert Binary Tree
        
        Time: O(n), Space: O(h) for recursion
        """
        if not root:
            return None
        
        # Swap left and right children
        root.left, root.right = root.right, root.left
        
        # Recursively invert subtrees
        self.invert_tree(root.left)
        self.invert_tree(root.right)
        
        return root
    
    def prune_tree(self, root: Optional[TreeNode], threshold: int) -> Optional[TreeNode]:
        """
        Prune tree by removing subtrees where all nodes are below threshold
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return None
        
        # Recursively prune left and right subtrees
        root.left = self.prune_tree(root.left, threshold)
        root.right = self.prune_tree(root.right, threshold)
        
        # If current node is below threshold and has no children, remove it
        if root.val < threshold and not root.left and not root.right:
            return None
        
        return root
    
    def copy_tree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """Create deep copy of tree"""
        if not root:
            return None
        
        new_root = TreeNode(root.val)
        new_root.left = self.copy_tree(root.left)
        new_root.right = self.copy_tree(root.right)
        
        return new_root
    
    def display_flattened(self, root: Optional[TreeNode]) -> None:
        """Display flattened tree as linked list"""
        values = []
        current = root
        
        while current:
            values.append(str(current.val))
            current = current.right
        
        print(f"     {' -> '.join(values)} -> null")
    
    def display_tree_structure(self, root: Optional[TreeNode]) -> None:
        """Display tree structure using level order"""
        if not root:
            print("     (empty tree)")
            return
        
        queue = deque([root])
        level = 0
        
        while queue:
            level_size = len(queue)
            level_values = []
            has_next_level = False
            
            for _ in range(level_size):
                node = queue.popleft()
                if node:
                    level_values.append(str(node.val))
                    queue.append(node.left)
                    queue.append(node.right)
                    if node.left or node.right:
                        has_next_level = True
                else:
                    level_values.append("null")
                    queue.append(None)
                    queue.append(None)
            
            print(f"     Level {level}: {level_values}")
            level += 1
            
            if not has_next_level:
                break


# ==========================================
# 3. TREE PATH AND DISTANCE PROBLEMS
# ==========================================

class TreePathProblems:
    """
    Advanced tree path and distance calculation problems
    """
    
    def explain_path_patterns(self) -> None:
        """Explain tree path problem patterns"""
        print("=== TREE PATH PROBLEM PATTERNS ===")
        print("Finding paths, distances, and path-based calculations")
        print()
        print("PATH TYPES:")
        print("• Root-to-leaf paths: Start from root, end at leaf")
        print("• Node-to-node paths: Any node to any other node")
        print("• Paths with specific sum: Target sum constraints")
        print("• Longest/shortest paths: Optimization problems")
        print("• Paths with specific properties: Even/odd nodes, etc.")
        print()
        print("SOLUTION STRATEGIES:")
        print("• DFS with backtracking for path enumeration")
        print("• Two-pass DFS for diameter calculations")
        print("• Path sum tracking with prefix sums")
        print("• LCA for shortest path between nodes")
        print("• Dynamic programming on trees")
        print()
        print("COMMON TECHNIQUES:")
        print("• Pass path state through recursion parameters")
        print("• Use global variables for cross-subtree calculations")
        print("• Maintain path history for backtracking")
        print("• Consider paths passing through each node as root")
    
    def demonstrate_path_problems(self) -> None:
        """Demonstrate advanced path problems"""
        print("=== TREE PATH PROBLEM DEMONSTRATIONS ===")
        
        path_solver = TreePathSolver()
        
        # Create sample tree with values
        root = TreeNode(10)
        root.left = TreeNode(5)
        root.right = TreeNode(-3)
        root.left.left = TreeNode(3)
        root.left.right = TreeNode(2)
        root.right.right = TreeNode(11)
        root.left.left.left = TreeNode(3)
        root.left.left.right = TreeNode(-2)
        root.left.right.right = TreeNode(1)
        
        print("Sample tree:")
        print("        10")
        print("       /  \\")
        print("      5   -3")
        print("     / \\    \\")
        print("    3   2    11")
        print("   / \\   \\")
        print("  3  -2   1")
        print()
        
        # Problem 1: Binary Tree Maximum Path Sum
        print("1. BINARY TREE MAXIMUM PATH SUM")
        max_sum = path_solver.max_path_sum(root)
        print(f"   Maximum path sum: {max_sum}")
        print("   (Path can be any node-to-node path)")
        print()
        
        # Problem 2: Path Sum III
        print("2. PATH SUM III - Count paths with target sum")
        target_sum = 8
        count = path_solver.path_sum_count(root, target_sum)
        print(f"   Target sum: {target_sum}")
        print(f"   Number of paths with sum {target_sum}: {count}")
        print()
        
        # Problem 3: Tree Diameter
        print("3. DIAMETER OF BINARY TREE")
        diameter = path_solver.diameter_of_tree(root)
        print(f"   Diameter (longest path between any two nodes): {diameter}")
        print()
        
        # Problem 4: All Root-to-Leaf Paths
        print("4. ALL ROOT-TO-LEAF PATHS")
        all_paths = path_solver.all_root_to_leaf_paths(root)
        print("   All root-to-leaf paths:")
        for i, path in enumerate(all_paths):
            print(f"     Path {i+1}: {' -> '.join(map(str, path))}")


class TreePathSolver:
    """
    Implementation of advanced tree path algorithms
    """
    
    def __init__(self):
        self.max_sum = float('-inf')
        self.diameter = 0
    
    def max_path_sum(self, root: Optional[TreeNode]) -> int:
        """
        LeetCode 124: Binary Tree Maximum Path Sum
        
        Time: O(n), Space: O(h)
        """
        self.max_sum = float('-inf')
        
        def max_gain(node):
            if not node:
                return 0
            
            # Maximum gain from left and right subtrees
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)
            
            # Maximum path sum passing through current node
            current_max = node.val + left_gain + right_gain
            
            # Update global maximum
            self.max_sum = max(self.max_sum, current_max)
            
            # Return maximum gain when starting from current node
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return self.max_sum
    
    def path_sum_count(self, root: Optional[TreeNode], target_sum: int) -> int:
        """
        LeetCode 437: Path Sum III
        
        Time: O(n), Space: O(n)
        """
        def dfs(node, current_sum, prefix_sums):
            if not node:
                return 0
            
            current_sum += node.val
            
            # Number of paths ending at current node with target sum
            count = prefix_sums.get(current_sum - target_sum, 0)
            
            # Add current sum to prefix sums
            prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1
            
            # Recurse to children
            count += dfs(node.left, current_sum, prefix_sums)
            count += dfs(node.right, current_sum, prefix_sums)
            
            # Backtrack: remove current sum from prefix sums
            prefix_sums[current_sum] -= 1
            if prefix_sums[current_sum] == 0:
                del prefix_sums[current_sum]
            
            return count
        
        return dfs(root, 0, {0: 1})
    
    def diameter_of_tree(self, root: Optional[TreeNode]) -> int:
        """
        LeetCode 543: Diameter of Binary Tree
        
        Time: O(n), Space: O(h)
        """
        self.diameter = 0
        
        def max_depth(node):
            if not node:
                return 0
            
            left_depth = max_depth(node.left)
            right_depth = max_depth(node.right)
            
            # Diameter passing through current node
            self.diameter = max(self.diameter, left_depth + right_depth)
            
            return 1 + max(left_depth, right_depth)
        
        max_depth(root)
        return self.diameter
    
    def all_root_to_leaf_paths(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Find all root-to-leaf paths
        
        Time: O(n * h), Space: O(n * h)
        """
        paths = []
        
        def dfs(node, current_path):
            if not node:
                return
            
            current_path.append(node.val)
            
            # If leaf node, add path to results
            if not node.left and not node.right:
                paths.append(current_path[:])  # Make a copy
            else:
                # Continue to children
                dfs(node.left, current_path)
                dfs(node.right, current_path)
            
            # Backtrack
            current_path.pop()
        
        dfs(root, [])
        return paths


# ==========================================
# 4. ADVANCED BST PROBLEMS
# ==========================================

class AdvancedBSTProblems:
    """
    Complex Binary Search Tree problems
    """
    
    def explain_bst_patterns(self) -> None:
        """Explain advanced BST problem patterns"""
        print("=== ADVANCED BST PROBLEM PATTERNS ===")
        print("Complex operations on Binary Search Trees")
        print()
        print("BST VALIDATION:")
        print("• Check BST property with proper bounds")
        print("• Handle duplicate values based on problem constraints")
        print("• Validate with inorder traversal (should be sorted)")
        print("• Consider integer overflow in bound checking")
        print()
        print("BST CONSTRUCTION AND MODIFICATION:")
        print("• Insert/delete while maintaining BST property")
        print("• Balance BST to minimize height")
        print("• Convert to other data structures (DLL, array)")
        print("• Merge multiple BSTs efficiently")
        print()
        print("RANGE OPERATIONS:")
        print("• Range sum queries in BST")
        print("• Count nodes in given range")
        print("• Delete nodes in range")
        print("• Closest value to target")
        print()
        print("ADVANCED QUERIES:")
        print("• Kth smallest/largest element")
        print("• Lowest Common Ancestor in BST")
        print("• Two Sum in BST")
        print("• Serialize and deserialize BST")
    
    def demonstrate_bst_problems(self) -> None:
        """Demonstrate advanced BST problems"""
        print("=== ADVANCED BST DEMONSTRATIONS ===")
        
        bst_solver = BSTSolver()
        
        # Create sample BST
        root = TreeNode(6)
        root.left = TreeNode(2)
        root.right = TreeNode(8)
        root.left.left = TreeNode(0)
        root.left.right = TreeNode(4)
        root.right.left = TreeNode(7)
        root.right.right = TreeNode(9)
        root.left.right.left = TreeNode(3)
        root.left.right.right = TreeNode(5)
        
        print("Sample BST:")
        print("        6")
        print("      /   \\")
        print("     2     8")
        print("   / \\   / \\")
        print("  0   4 7   9")
        print("     / \\")
        print("    3   5")
        print()
        
        # Problem 1: Validate BST
        print("1. VALIDATE BINARY SEARCH TREE")
        is_valid = bst_solver.is_valid_bst(root)
        print(f"   Is valid BST: {is_valid}")
        print()
        
        # Problem 2: Kth smallest element
        print("2. KTH SMALLEST ELEMENT IN BST")
        k = 3
        kth_smallest = bst_solver.kth_smallest(root, k)
        print(f"   {k}rd smallest element: {kth_smallest}")
        print()
        
        # Problem 3: Range sum
        print("3. RANGE SUM OF BST")
        low, high = 7, 15
        range_sum = bst_solver.range_sum_bst(root, low, high)
        print(f"   Sum of nodes in range [{low}, {high}]: {range_sum}")
        print()
        
        # Problem 4: Convert BST to sorted DLL
        print("4. CONVERT BST TO SORTED DOUBLY LINKED LIST")
        print("   Converting BST to sorted doubly linked list...")
        
        # Make a copy for conversion
        root_copy = bst_solver.copy_tree(root)
        dll_head = bst_solver.tree_to_doubly_list(root_copy)
        
        print("   Doubly linked list (forward):")
        bst_solver.display_dll_forward(dll_head)


class BSTSolver:
    """
    Advanced BST problem solutions
    """
    
    def is_valid_bst(self, root: Optional[TreeNode]) -> bool:
        """
        LeetCode 98: Validate Binary Search Tree
        
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
    
    def kth_smallest(self, root: Optional[TreeNode], k: int) -> int:
        """
        LeetCode 230: Kth Smallest Element in a BST
        
        Time: O(h + k), Space: O(h)
        """
        def inorder(node):
            if not node:
                return []
            return inorder(node.left) + [node.val] + inorder(node.right)
        
        # Alternative: iterative approach with early termination
        stack = []
        current = root
        count = 0
        
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            
            current = stack.pop()
            count += 1
            
            if count == k:
                return current.val
            
            current = current.right
        
        return -1
    
    def range_sum_bst(self, root: Optional[TreeNode], low: int, high: int) -> int:
        """
        LeetCode 938: Range Sum of BST
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return 0
        
        total = 0
        
        # Include current node if in range
        if low <= root.val <= high:
            total += root.val
        
        # Recurse left only if current val > low
        if root.val > low:
            total += self.range_sum_bst(root.left, low, high)
        
        # Recurse right only if current val < high
        if root.val < high:
            total += self.range_sum_bst(root.right, low, high)
        
        return total
    
    def tree_to_doubly_list(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Convert BST to sorted circular doubly linked list
        
        Time: O(n), Space: O(h)
        """
        if not root:
            return None
        
        self.first = None
        self.last = None
        
        def inorder(node):
            if not node:
                return
            
            # Process left subtree
            inorder(node.left)
            
            # Process current node
            if self.last:
                # Link previous node with current
                self.last.right = node
                node.left = self.last
            else:
                # First node
                self.first = node
            
            self.last = node
            
            # Process right subtree
            inorder(node.right)
        
        inorder(root)
        
        # Make it circular
        if self.first and self.last:
            self.last.right = self.first
            self.first.left = self.last
        
        return self.first
    
    def copy_tree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """Create deep copy of tree"""
        if not root:
            return None
        
        new_root = TreeNode(root.val)
        new_root.left = self.copy_tree(root.left)
        new_root.right = self.copy_tree(root.right)
        
        return new_root
    
    def display_dll_forward(self, head: Optional[TreeNode]) -> None:
        """Display doubly linked list in forward direction"""
        if not head:
            print("     (empty list)")
            return
        
        values = []
        current = head
        
        # Traverse until we come back to head (circular)
        while True:
            values.append(str(current.val))
            current = current.right
            if current == head:
                break
        
        print(f"     {' <-> '.join(values)} (circular)")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_tree_interview_problems():
    """Demonstrate all tree interview problems"""
    print("=== TREE INTERVIEW PROBLEMS COMPREHENSIVE GUIDE ===\n")
    
    problems = TreeInterviewProblems()
    
    # 1. Tree construction problems
    problems.explain_construction_patterns()
    print("\n" + "="*60 + "\n")
    
    problems.demonstrate_construction_problems()
    print("\n" + "="*60 + "\n")
    
    # 2. Tree transformation problems
    transformer = TreeTransformation()
    transformer.explain_transformation_patterns()
    print("\n" + "="*60 + "\n")
    
    transformer.demonstrate_transformation_problems()
    print("\n" + "="*60 + "\n")
    
    # 3. Tree path problems
    path_problems = TreePathProblems()
    path_problems.explain_path_patterns()
    print("\n" + "="*60 + "\n")
    
    path_problems.demonstrate_path_problems()
    print("\n" + "="*60 + "\n")
    
    # 4. Advanced BST problems
    bst_problems = AdvancedBSTProblems()
    bst_problems.explain_bst_patterns()
    print("\n" + "="*60 + "\n")
    
    bst_problems.demonstrate_bst_problems()


if __name__ == "__main__":
    demonstrate_tree_interview_problems()
    
    print("\n=== TREE INTERVIEW MASTERY STRATEGY ===")
    
    print("\n🎯 KEY PROBLEM CATEGORIES:")
    print("• Tree Construction: Build trees from various inputs")
    print("• Tree Transformation: Modify structure while preserving properties")
    print("• Path Problems: Find paths with specific constraints")
    print("• BST Operations: Leverage BST properties for efficient solutions")
    print("• Tree Validation: Verify tree properties and constraints")
    
    print("\n📊 COMPLEXITY PATTERNS:")
    print("• Most tree problems: O(n) time, O(h) space for recursion")
    print("• Path counting: O(n) time, O(n) space with prefix sums")
    print("• BST operations: O(h) time on average, O(n) worst case")
    print("• Tree construction: O(n) time, O(n) space typically")
    print("• Transformation: O(n) time, O(1) to O(n) space depending on approach")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Use iterative approaches to reduce space complexity")
    print("• Leverage BST properties for pruning search space")
    print("• Apply memoization for overlapping subproblems")
    print("• Use Morris traversal for O(1) space when possible")
    print("• Consider two-pass algorithms for global optimizations")
    
    print("\n🔧 INTERVIEW TECHNIQUES:")
    print("• Always clarify tree structure and constraints")
    print("• Start with recursive solution, then optimize if needed")
    print("• Handle edge cases: empty tree, single node, skewed tree")
    print("• Use proper bounds checking for BST problems")
    print("• Consider both top-down and bottom-up approaches")
    
    print("\n🏆 MUST-KNOW PROBLEMS:")
    print("• LeetCode 105: Construct Binary Tree from Preorder and Inorder")
    print("• LeetCode 124: Binary Tree Maximum Path Sum")
    print("• LeetCode 98: Validate Binary Search Tree")
    print("• LeetCode 114: Flatten Binary Tree to Linked List")
    print("• LeetCode 437: Path Sum III")
    print("• LeetCode 226: Invert Binary Tree")
    print("• LeetCode 543: Diameter of Binary Tree")
    print("• LeetCode 230: Kth Smallest Element in BST")
    
    print("\n🎓 PREPARATION STRATEGY:")
    print("1. Master basic tree traversals and properties")
    print("2. Practice construction from different inputs")
    print("3. Learn path-based problem patterns")
    print("4. Study BST-specific optimizations")
    print("5. Practice with time constraints")
    print("6. Focus on clean, bug-free implementations")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Draw tree diagrams for complex problems")
    print("• Test with various tree shapes (balanced, skewed, single node)")
    print("• Practice explaining approach before coding")
    print("• Master both recursive and iterative solutions")
    print("• Understand when to use global variables vs return values")
    print("• Learn to identify and handle base cases correctly")
