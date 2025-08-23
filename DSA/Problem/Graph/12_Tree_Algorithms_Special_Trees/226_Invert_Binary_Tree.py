"""
226. Invert Binary Tree - Multiple Approaches
Difficulty: Easy

Given the root of a binary tree, invert the tree, and return its root.

Inverting a binary tree means swapping the left and right children of every node.
"""

from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class InvertBinaryTree:
    """Multiple approaches to invert binary tree"""
    
    def invertTree_recursive(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 1: Recursive Inversion
        
        Classic recursive approach - swap children and recurse.
        
        Time: O(n) - visit each node once
        Space: O(h) - recursion stack
        """
        if not root:
            return None
        
        # Swap left and right children
        root.left, root.right = root.right, root.left
        
        # Recursively invert subtrees
        self.invertTree_recursive(root.left)
        self.invertTree_recursive(root.right)
        
        return root
    
    def invertTree_iterative_dfs(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 2: Iterative DFS using Stack
        
        Use explicit stack to avoid recursion.
        
        Time: O(n)
        Space: O(h) - stack space
        """
        if not root:
            return None
        
        stack = [root]
        
        while stack:
            node = stack.pop()
            
            # Swap children
            node.left, node.right = node.right, node.left
            
            # Add children to stack
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        
        return root
    
    def invertTree_iterative_bfs(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 3: Iterative BFS using Queue
        
        Level-order traversal with inversion.
        
        Time: O(n)
        Space: O(w) - queue width
        """
        if not root:
            return None
        
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            
            # Swap children
            node.left, node.right = node.right, node.left
            
            # Add children to queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return root
    
    def invertTree_postorder_iterative(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 4: Postorder Iterative Traversal
        
        Process children before parent using postorder.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return None
        
        stack = []
        current = root
        last_visited = None
        
        while stack or current:
            if current:
                stack.append(current)
                current = current.left
            else:
                peek_node = stack[-1]
                
                if peek_node.right and last_visited != peek_node.right:
                    current = peek_node.right
                else:
                    node = stack.pop()
                    
                    # Swap children (postorder processing)
                    node.left, node.right = node.right, node.left
                    
                    last_visited = node
        
        return root
    
    def invertTree_morris_traversal(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 5: Morris Traversal (Space Optimized)
        
        Use Morris traversal for O(1) space complexity.
        
        Time: O(n)
        Space: O(1)
        """
        if not root:
            return None
        
        current = root
        
        while current:
            if not current.left:
                # Swap children and move to right
                current.left, current.right = current.right, current.left
                current = current.left  # This is now the original right child
            else:
                # Find inorder predecessor
                predecessor = current.left
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Make threading connection
                    predecessor.right = current
                    current = current.left
                else:
                    # Remove threading connection and swap
                    predecessor.right = None
                    current.left, current.right = current.right, current.left
                    current = current.left  # Move to original right child
        
        return root
    
    def invertTree_level_by_level(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 6: Level-by-Level Processing
        
        Process each level completely before moving to next.
        
        Time: O(n)
        Space: O(w)
        """
        if not root:
            return None
        
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            
            # Process entire level
            for _ in range(level_size):
                node = queue.popleft()
                
                # Swap children
                node.left, node.right = node.right, node.left
                
                # Add next level nodes
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        return root
    
    def invertTree_divide_conquer(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 7: Divide and Conquer
        
        Explicitly use divide and conquer paradigm.
        
        Time: O(n)
        Space: O(h)
        """
        def divide_conquer(node):
            # Base case
            if not node:
                return None
            
            # Divide: recursively invert subtrees
            left_inverted = divide_conquer(node.left)
            right_inverted = divide_conquer(node.right)
            
            # Conquer: swap the inverted subtrees
            node.left = right_inverted
            node.right = left_inverted
            
            return node
        
        return divide_conquer(root)
    
    def invertTree_with_path_tracking(self, root: Optional[TreeNode]) -> tuple:
        """
        Approach 8: Invert with Path Tracking
        
        Track the inversion process for debugging/analysis.
        
        Time: O(n)
        Space: O(h + n) - for path tracking
        """
        if not root:
            return None, []
        
        inversion_log = []
        
        def invert_with_log(node, path):
            if not node:
                return None
            
            # Log the inversion
            inversion_log.append({
                'node': node.val,
                'path': path[:],
                'original_left': node.left.val if node.left else None,
                'original_right': node.right.val if node.right else None
            })
            
            # Swap children
            node.left, node.right = node.right, node.left
            
            # Recursively invert with updated paths
            invert_with_log(node.left, path + ['L'])
            invert_with_log(node.right, path + ['R'])
            
            return node
        
        inverted_root = invert_with_log(root, [])
        return inverted_root, inversion_log

def create_test_trees():
    """Create test trees for inversion"""
    # Tree 1: [4,2,7,1,3,6,9]
    tree1 = TreeNode(4)
    tree1.left = TreeNode(2)
    tree1.right = TreeNode(7)
    tree1.left.left = TreeNode(1)
    tree1.left.right = TreeNode(3)
    tree1.right.left = TreeNode(6)
    tree1.right.right = TreeNode(9)
    
    # Tree 2: [2,1,3]
    tree2 = TreeNode(2)
    tree2.left = TreeNode(1)
    tree2.right = TreeNode(3)
    
    # Tree 3: Single node
    tree3 = TreeNode(1)
    
    # Tree 4: Left skewed
    tree4 = TreeNode(1)
    tree4.left = TreeNode(2)
    tree4.left.left = TreeNode(3)
    
    return [
        (tree1, "Balanced tree"),
        (tree2, "Simple tree"),
        (tree3, "Single node"),
        (tree4, "Left skewed"),
        (None, "Empty tree")
    ]

def tree_to_list_level_order(root: Optional[TreeNode]) -> List:
    """Convert tree to level-order list for comparison"""
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

def copy_tree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """Create a deep copy of the tree"""
    if not root:
        return None
    
    new_root = TreeNode(root.val)
    new_root.left = copy_tree(root.left)
    new_root.right = copy_tree(root.right)
    
    return new_root

def test_invert_algorithms():
    """Test all tree inversion algorithms"""
    solver = InvertBinaryTree()
    test_trees = create_test_trees()
    
    algorithms = [
        ("Recursive", solver.invertTree_recursive),
        ("Iterative DFS", solver.invertTree_iterative_dfs),
        ("Iterative BFS", solver.invertTree_iterative_bfs),
        ("Postorder", solver.invertTree_postorder_iterative),
        ("Morris Traversal", solver.invertTree_morris_traversal),
        ("Level by Level", solver.invertTree_level_by_level),
        ("Divide & Conquer", solver.invertTree_divide_conquer),
    ]
    
    print("=== Testing Tree Inversion Algorithms ===")
    
    for original_tree, description in test_trees:
        print(f"\n--- {description} ---")
        
        if original_tree:
            original_list = tree_to_list_level_order(original_tree)
            print(f"Original: {original_list}")
        else:
            print("Original: []")
        
        for alg_name, alg_func in algorithms:
            try:
                # Create copy for each algorithm
                tree_copy = copy_tree(original_tree)
                inverted = alg_func(tree_copy)
                inverted_list = tree_to_list_level_order(inverted)
                
                print(f"{alg_name:15} | Result: {inverted_list}")
                
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")
    
    # Test path tracking
    print(f"\n--- Inversion Path Tracking Demo ---")
    original_tree = create_test_trees()[0][0]  # Use balanced tree
    tree_copy = copy_tree(original_tree)
    inverted_root, log = solver.invertTree_with_path_tracking(tree_copy)
    
    print("Inversion log:")
    for entry in log[:5]:  # Show first 5 entries
        print(f"  Node {entry['node']} at path {entry['path']}: "
              f"L={entry['original_left']} -> R={entry['original_right']}")

def demonstrate_inversion_concepts():
    """Demonstrate tree inversion concepts"""
    print("\n=== Tree Inversion Concepts ===")
    
    print("Inversion Definition:")
    print("• Swap left and right children of every node")
    print("• Recursive operation applied to entire tree")
    print("• Also known as 'mirroring' the tree")
    print("• Preserves tree structure, changes orientation")
    
    print("\nInversion Properties:")
    print("• Inverting twice returns original tree")
    print("• Inorder traversal becomes reverse inorder")
    print("• Preorder becomes modified postorder")
    print("• Tree height and node count unchanged")
    
    print("\nTraversal Changes:")
    print("• Original inorder: Left -> Root -> Right")
    print("• Inverted inorder: Right -> Root -> Left")
    print("• Preorder and postorder also affected")
    print("• Level-order structure changes")
    
    print("\nApplications:")
    print("• Image processing (horizontal flip)")
    print("• Computer graphics transformations")
    print("• Tree structure analysis")
    print("• Algorithm testing and validation")

def analyze_inversion_complexity():
    """Analyze complexity of different inversion approaches"""
    print("\n=== Complexity Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Recursive (Recommended):**")
    print("   • Time: O(n) - visit each node once")
    print("   • Space: O(h) - recursion stack")
    print("   • Pros: Simple, intuitive")
    print("   • Cons: Stack overflow for deep trees")
    
    print("\n2. **Iterative DFS:**")
    print("   • Time: O(n)")
    print("   • Space: O(h) - explicit stack")
    print("   • Pros: No recursion limits")
    print("   • Cons: Slightly more complex")
    
    print("\n3. **Iterative BFS:**")
    print("   • Time: O(n)")
    print("   • Space: O(w) - queue width")
    print("   • Pros: Level-by-level processing")
    print("   • Cons: May use more space for wide trees")
    
    print("\n4. **Postorder Iterative:**")
    print("   • Time: O(n)")
    print("   • Space: O(h)")
    print("   • Pros: Bottom-up processing")
    print("   • Cons: More complex implementation")
    
    print("\n5. **Morris Traversal:**")
    print("   • Time: O(n)")
    print("   • Space: O(1) - constant space")
    print("   • Pros: Optimal space complexity")
    print("   • Cons: Complex, modifies tree temporarily")
    
    print("\n6. **Level by Level:**")
    print("   • Time: O(n)")
    print("   • Space: O(w)")
    print("   • Pros: Clear level processing")
    print("   • Cons: Similar to BFS")
    
    print("\nRecommendation: Recursive approach for simplicity, Iterative DFS for deep trees")

def demonstrate_inversion_applications():
    """Demonstrate practical applications of tree inversion"""
    print("\n=== Practical Applications ===")
    
    print("Real-World Uses:")
    
    print("\n1. **Computer Graphics:**")
    print("   • Horizontal flipping of images")
    print("   • Mirror transformations")
    print("   • Scene graph manipulations")
    print("   • UI layout mirroring (RTL languages)")
    
    print("\n2. **Data Structure Operations:**")
    print("   • Tree structure testing")
    print("   • Symmetry detection")
    print("   • Tree comparison algorithms")
    print("   • Canonical form generation")
    
    print("\n3. **Algorithm Design:**")
    print("   • Divide and conquer problems")
    print("   • Tree traversal variations")
    print("   • Pattern matching in trees")
    print("   • Optimization problems")
    
    print("\n4. **Game Development:**")
    print("   • Minimax tree inversion")
    print("   • Game state transformations")
    print("   • AI decision tree modifications")
    print("   • Procedural generation")
    
    print("\n5. **Educational Purposes:**")
    print("   • Teaching recursion concepts")
    print("   • Tree traversal understanding")
    print("   • Algorithm complexity analysis")
    print("   • Data structure manipulation")

def demonstrate_inversion_variations():
    """Demonstrate variations of tree inversion"""
    print("\n=== Inversion Variations ===")
    
    print("Problem Variations:")
    
    print("\n1. **Conditional Inversion:**")
    print("   • Invert only nodes satisfying certain conditions")
    print("   • Example: Invert only nodes with even values")
    print("   • Requires additional logic in traversal")
    
    print("\n2. **Partial Inversion:**")
    print("   • Invert only specific subtrees")
    print("   • Based on node properties or positions")
    print("   • More complex decision making")
    
    print("\n3. **Level-wise Inversion:**")
    print("   • Invert alternate levels only")
    print("   • Creates interesting tree patterns")
    print("   • Useful in certain algorithms")
    
    print("\n4. **Weighted Inversion:**")
    print("   • Consider edge weights in inversion")
    print("   • May affect inversion decisions")
    print("   • Applications in network problems")
    
    print("\n5. **Multi-way Tree Inversion:**")
    print("   • Extend to trees with more than 2 children")
    print("   • Reverse order of all children")
    print("   • More complex but similar principles")

if __name__ == "__main__":
    test_invert_algorithms()
    demonstrate_inversion_concepts()
    analyze_inversion_complexity()
    demonstrate_inversion_applications()
    demonstrate_inversion_variations()

"""
Invert Binary Tree - Key Insights:

1. **Problem Essence:**
   - Swap left and right children of every node
   - Simple but fundamental tree operation
   - Demonstrates basic tree manipulation
   - Foundation for more complex algorithms

2. **Algorithm Strategies:**
   - Recursive: Natural tree recursion
   - Iterative: DFS or BFS with explicit data structures
   - Space-optimized: Morris traversal technique
   - Specialized: Postorder, level-by-level processing

3. **Implementation Approaches:**
   - Top-down: Swap then recurse on children
   - Bottom-up: Recurse on children then swap
   - Iterative: Use stack or queue for traversal
   - In-place: Modify tree structure directly

4. **Complexity Considerations:**
   - Time: O(n) for all approaches - must visit each node
   - Space: Varies from O(1) to O(h) or O(w)
   - Recursive: Simple but limited by stack depth
   - Iterative: More control over space usage

5. **Practical Applications:**
   - Computer graphics transformations
   - Tree structure testing and validation
   - Algorithm design and optimization
   - Educational tool for recursion concepts

The inversion problem showcases fundamental tree traversal
techniques and serves as building block for complex algorithms.
"""
