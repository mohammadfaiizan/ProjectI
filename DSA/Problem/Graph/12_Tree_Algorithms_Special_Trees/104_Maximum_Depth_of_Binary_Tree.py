"""
104. Maximum Depth of Binary Tree - Multiple Approaches
Difficulty: Easy

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path 
from the root node down to the farthest leaf node.
"""

from typing import Optional, List, Tuple
from collections import deque
import sys

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class MaxDepthBinaryTree:
    """Multiple approaches to find maximum depth of binary tree"""
    
    def maxDepth_recursive_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 1: Recursive DFS (Top-down)
        
        Classic recursive approach using depth-first search.
        
        Time: O(n) - visit each node once
        Space: O(h) - recursion stack, h is height of tree
        """
        if not root:
            return 0
        
        left_depth = self.maxDepth_recursive_dfs(root.left)
        right_depth = self.maxDepth_recursive_dfs(root.right)
        
        return 1 + max(left_depth, right_depth)
    
    def maxDepth_iterative_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 2: Iterative DFS using Stack
        
        Use explicit stack to avoid recursion and potential stack overflow.
        
        Time: O(n)
        Space: O(h) - explicit stack
        """
        if not root:
            return 0
        
        stack = [(root, 1)]  # (node, current_depth)
        max_depth = 0
        
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            
            if node.left:
                stack.append((node.left, depth + 1))
            if node.right:
                stack.append((node.right, depth + 1))
        
        return max_depth
    
    def maxDepth_bfs_level_order(self, root: Optional[TreeNode]) -> int:
        """
        Approach 3: BFS Level Order Traversal
        
        Process tree level by level using queue.
        
        Time: O(n)
        Space: O(w) - w is maximum width of tree
        """
        if not root:
            return 0
        
        queue = deque([root])
        depth = 0
        
        while queue:
            depth += 1
            level_size = len(queue)
            
            # Process all nodes at current level
            for _ in range(level_size):
                node = queue.popleft()
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return depth
    
    def maxDepth_morris_traversal(self, root: Optional[TreeNode]) -> int:
        """
        Approach 4: Morris Traversal (Space Optimized)
        
        Use Morris traversal technique for O(1) space complexity.
        
        Time: O(n)
        Space: O(1) - no recursion or explicit stack
        """
        if not root:
            return 0
        
        max_depth = 0
        current = root
        depth = 0
        
        while current:
            if not current.left:
                # No left child, go to right
                depth += 1
                max_depth = max(max_depth, depth)
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                temp_depth = depth + 1
                
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                    temp_depth += 1
                
                if not predecessor.right:
                    # Make threading connection
                    predecessor.right = current
                    depth += 1
                    current = current.left
                else:
                    # Remove threading connection
                    predecessor.right = None
                    max_depth = max(max_depth, temp_depth)
                    current = current.right
        
        return max_depth
    
    def maxDepth_postorder_iterative(self, root: Optional[TreeNode]) -> int:
        """
        Approach 5: Iterative Postorder Traversal
        
        Use postorder traversal to calculate depth bottom-up.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return 0
        
        stack = []
        node_depth = {}
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
                    # Process current node
                    node = stack.pop()
                    
                    left_depth = node_depth.get(node.left, 0)
                    right_depth = node_depth.get(node.right, 0)
                    node_depth[node] = 1 + max(left_depth, right_depth)
                    
                    last_visited = node
        
        return node_depth.get(root, 0)
    
    def maxDepth_divide_conquer(self, root: Optional[TreeNode]) -> int:
        """
        Approach 6: Divide and Conquer
        
        Explicitly use divide and conquer paradigm.
        
        Time: O(n)
        Space: O(h)
        """
        def divide_conquer(node):
            # Base case
            if not node:
                return 0
            
            # Divide
            left_result = divide_conquer(node.left)
            right_result = divide_conquer(node.right)
            
            # Conquer
            return 1 + max(left_result, right_result)
        
        return divide_conquer(root)
    
    def maxDepth_with_path_tracking(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Approach 7: Track Path to Deepest Node
        
        Return both maximum depth and the path to deepest node.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return 0, []
        
        max_depth = 0
        deepest_path = []
        
        def dfs(node, current_path, depth):
            nonlocal max_depth, deepest_path
            
            if not node:
                return
            
            current_path.append(node.val)
            
            if depth > max_depth:
                max_depth = depth
                deepest_path = current_path[:]
            
            dfs(node.left, current_path, depth + 1)
            dfs(node.right, current_path, depth + 1)
            
            current_path.pop()
        
        dfs(root, [], 1)
        return max_depth, deepest_path
    
    def maxDepth_optimized_early_termination(self, root: Optional[TreeNode]) -> int:
        """
        Approach 8: Optimized with Early Termination
        
        Use pruning techniques for potential optimization.
        
        Time: O(n) worst case, better average case
        Space: O(h)
        """
        if not root:
            return 0
        
        def dfs_with_pruning(node, current_depth, current_max):
            if not node:
                return current_max
            
            current_max = max(current_max, current_depth)
            
            # Early termination if we can't improve
            remaining_levels = sys.getrecursionlimit() - current_depth
            if current_max + remaining_levels <= current_max:
                return current_max
            
            left_max = dfs_with_pruning(node.left, current_depth + 1, current_max)
            right_max = dfs_with_pruning(node.right, current_depth + 1, current_max)
            
            return max(left_max, right_max)
        
        return dfs_with_pruning(root, 1, 0)

def create_test_trees():
    """Create test binary trees for testing"""
    # Tree 1: [3,9,20,null,null,15,7] - depth 3
    tree1 = TreeNode(3)
    tree1.left = TreeNode(9)
    tree1.right = TreeNode(20)
    tree1.right.left = TreeNode(15)
    tree1.right.right = TreeNode(7)
    
    # Tree 2: [1,null,2] - depth 2
    tree2 = TreeNode(1)
    tree2.right = TreeNode(2)
    
    # Tree 3: Single node - depth 1
    tree3 = TreeNode(1)
    
    # Tree 4: Empty tree - depth 0
    tree4 = None
    
    # Tree 5: Deep left skewed tree - depth 4
    tree5 = TreeNode(1)
    tree5.left = TreeNode(2)
    tree5.left.left = TreeNode(3)
    tree5.left.left.left = TreeNode(4)
    
    return [
        (tree1, 3, "Balanced tree"),
        (tree2, 2, "Right skewed"),
        (tree3, 1, "Single node"),
        (tree4, 0, "Empty tree"),
        (tree5, 4, "Left skewed")
    ]

def test_max_depth_algorithms():
    """Test all maximum depth algorithms"""
    solver = MaxDepthBinaryTree()
    
    algorithms = [
        ("Recursive DFS", solver.maxDepth_recursive_dfs),
        ("Iterative DFS", solver.maxDepth_iterative_dfs),
        ("BFS Level Order", solver.maxDepth_bfs_level_order),
        ("Morris Traversal", solver.maxDepth_morris_traversal),
        ("Postorder Iterative", solver.maxDepth_postorder_iterative),
        ("Divide & Conquer", solver.maxDepth_divide_conquer),
        ("Early Termination", solver.maxDepth_optimized_early_termination),
    ]
    
    test_trees = create_test_trees()
    
    print("=== Testing Maximum Depth Algorithms ===")
    
    for tree, expected, description in test_trees:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(tree)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Depth: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")
    
    # Test path tracking
    print(f"\n--- Path Tracking Demo ---")
    tree, expected, description = test_trees[0]  # Use balanced tree
    depth, path = solver.maxDepth_with_path_tracking(tree)
    print(f"Max depth: {depth}, Path to deepest: {path}")

def demonstrate_tree_depth_concepts():
    """Demonstrate tree depth concepts and applications"""
    print("\n=== Tree Depth Concepts ===")
    
    print("Key Concepts:")
    print("• Depth/Height: Number of edges from root to farthest leaf")
    print("• Level: Depth of a node (root is level 0 or 1)")
    print("• Balanced tree: Height is O(log n)")
    print("• Skewed tree: Height can be O(n)")
    
    print("\nTraversal Methods:")
    print("• DFS (Recursive): Natural recursive solution")
    print("• DFS (Iterative): Explicit stack, avoids recursion limits")
    print("• BFS (Level Order): Process level by level")
    print("• Morris: O(1) space using threading")
    
    print("\nComplexity Analysis:")
    print("• Time: O(n) for all approaches - must visit each node")
    print("• Space: O(h) for recursive, O(w) for BFS, O(1) for Morris")
    print("• h = height, w = maximum width of tree")
    
    print("\nApplications:")
    print("• Tree validation and analysis")
    print("• Memory usage estimation")
    print("• Algorithm complexity analysis")
    print("• Tree balancing decisions")

def analyze_space_time_tradeoffs():
    """Analyze space-time tradeoffs of different approaches"""
    print("\n=== Space-Time Tradeoff Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Recursive DFS:**")
    print("   • Time: O(n) - visit each node once")
    print("   • Space: O(h) - recursion stack")
    print("   • Pros: Simple, intuitive")
    print("   • Cons: Stack overflow for deep trees")
    
    print("\n2. **Iterative DFS:**")
    print("   • Time: O(n)")
    print("   • Space: O(h) - explicit stack")
    print("   • Pros: No recursion limits")
    print("   • Cons: More complex implementation")
    
    print("\n3. **BFS Level Order:**")
    print("   • Time: O(n)")
    print("   • Space: O(w) - queue width")
    print("   • Pros: Natural level processing")
    print("   • Cons: Can use more space for wide trees")
    
    print("\n4. **Morris Traversal:**")
    print("   • Time: O(n)")
    print("   • Space: O(1) - constant space")
    print("   • Pros: Optimal space complexity")
    print("   • Cons: Complex implementation, modifies tree temporarily")
    
    print("\n5. **Postorder Iterative:**")
    print("   • Time: O(n)")
    print("   • Space: O(h)")
    print("   • Pros: Bottom-up calculation")
    print("   • Cons: More complex than recursive")
    
    print("\nRecommendations:")
    print("• General use: Recursive DFS (simple and efficient)")
    print("• Deep trees: Iterative DFS (avoid stack overflow)")
    print("• Memory constrained: Morris traversal")
    print("• Level analysis: BFS level order")

def demonstrate_tree_properties():
    """Demonstrate various tree properties related to depth"""
    print("\n=== Tree Properties and Depth ===")
    
    print("Tree Type Classifications:")
    
    print("\n1. **Complete Binary Tree:**")
    print("   • All levels filled except possibly last")
    print("   • Height: ⌊log₂(n)⌋")
    print("   • Optimal for heap operations")
    
    print("\n2. **Perfect Binary Tree:**")
    print("   • All internal nodes have 2 children")
    print("   • All leaves at same level")
    print("   • Height: log₂(n+1) - 1")
    
    print("\n3. **Balanced Binary Tree:**")
    print("   • Height difference between subtrees ≤ 1")
    print("   • Height: O(log n)")
    print("   • Examples: AVL, Red-Black trees")
    
    print("\n4. **Skewed Binary Tree:**")
    print("   • Each node has at most one child")
    print("   • Height: n - 1 (worst case)")
    print("   • Degenerates to linked list")
    
    print("\nDepth-Related Properties:")
    print("• Number of nodes at level k: 2^k")
    print("• Maximum nodes in tree of height h: 2^(h+1) - 1")
    print("• Minimum height for n nodes: ⌈log₂(n+1)⌉ - 1")
    print("• Maximum height for n nodes: n - 1")

if __name__ == "__main__":
    test_max_depth_algorithms()
    demonstrate_tree_depth_concepts()
    analyze_space_time_tradeoffs()
    demonstrate_tree_properties()

"""
Maximum Depth of Binary Tree - Key Insights:

1. **Problem Fundamentals:**
   - Tree depth = longest path from root to leaf
   - Must visit all nodes to find maximum
   - Multiple traversal strategies possible
   - Space-time tradeoffs in different approaches

2. **Algorithm Categories:**
   - Recursive: Natural tree recursion
   - Iterative: Explicit stack/queue management
   - Space-optimized: Morris traversal technique
   - Specialized: Path tracking, early termination

3. **Traversal Strategies:**
   - DFS: Depth-first exploration (recursive/iterative)
   - BFS: Level-by-level processing
   - Morris: Constant space using threading
   - Postorder: Bottom-up calculation

4. **Complexity Considerations:**
   - Time: O(n) for all approaches (must visit each node)
   - Space: Varies from O(1) to O(h) or O(w)
   - Balanced trees: O(log n) space
   - Skewed trees: O(n) space

5. **Practical Applications:**
   - Tree structure analysis
   - Memory usage estimation
   - Algorithm complexity bounds
   - Tree balancing decisions

The maximum depth problem demonstrates fundamental
tree traversal techniques and space-time optimization strategies.
"""
