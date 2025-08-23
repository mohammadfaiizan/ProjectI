"""
543. Diameter of Binary Tree - Multiple Approaches
Difficulty: Easy

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes 
in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.
"""

from typing import Optional, Tuple, List
from collections import defaultdict, deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class DiameterBinaryTree:
    """Multiple approaches to find diameter of binary tree"""
    
    def __init__(self):
        self.max_diameter = 0
    
    def diameterOfBinaryTree_recursive_optimized(self, root: Optional[TreeNode]) -> int:
        """
        Approach 1: Recursive with Single Pass
        
        Calculate diameter and height in single traversal.
        
        Time: O(n) - visit each node once
        Space: O(h) - recursion stack
        """
        self.max_diameter = 0
        
        def height_and_diameter(node):
            if not node:
                return 0
            
            left_height = height_and_diameter(node.left)
            right_height = height_and_diameter(node.right)
            
            # Diameter through current node
            current_diameter = left_height + right_height
            self.max_diameter = max(self.max_diameter, current_diameter)
            
            # Return height of current subtree
            return 1 + max(left_height, right_height)
        
        height_and_diameter(root)
        return self.max_diameter
    
    def diameterOfBinaryTree_two_pass(self, root: Optional[TreeNode]) -> int:
        """
        Approach 2: Two-Pass Algorithm
        
        First pass calculates heights, second pass finds diameter.
        
        Time: O(n^2) - height calculation for each node
        Space: O(h)
        """
        if not root:
            return 0
        
        def height(node):
            if not node:
                return 0
            return 1 + max(height(node.left), height(node.right))
        
        def diameter(node):
            if not node:
                return 0
            
            # Diameter through current node
            left_height = height(node.left)
            right_height = height(node.right)
            current_diameter = left_height + right_height
            
            # Diameter in left or right subtree
            left_diameter = diameter(node.left)
            right_diameter = diameter(node.right)
            
            return max(current_diameter, left_diameter, right_diameter)
        
        return diameter(root)
    
    def diameterOfBinaryTree_iterative_postorder(self, root: Optional[TreeNode]) -> int:
        """
        Approach 3: Iterative Postorder Traversal
        
        Use iterative postorder to calculate diameter bottom-up.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return 0
        
        stack = []
        heights = {}
        max_diameter = 0
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
                    
                    # Calculate height and diameter for current node
                    left_height = heights.get(node.left, 0)
                    right_height = heights.get(node.right, 0)
                    
                    heights[node] = 1 + max(left_height, right_height)
                    diameter_through_node = left_height + right_height
                    max_diameter = max(max_diameter, diameter_through_node)
                    
                    last_visited = node
        
        return max_diameter
    
    def diameterOfBinaryTree_bfs_with_heights(self, root: Optional[TreeNode]) -> int:
        """
        Approach 4: BFS with Height Calculation
        
        Use BFS to process nodes level by level, calculating heights.
        
        Time: O(n^2) in worst case
        Space: O(w) - width of tree
        """
        if not root:
            return 0
        
        def calculate_height(node):
            if not node:
                return 0
            
            queue = deque([(node, 1)])
            max_height = 0
            
            while queue:
                current, height = queue.popleft()
                max_height = max(max_height, height)
                
                if current.left:
                    queue.append((current.left, height + 1))
                if current.right:
                    queue.append((current.right, height + 1))
            
            return max_height
        
        queue = deque([root])
        max_diameter = 0
        
        while queue:
            node = queue.popleft()
            
            # Calculate diameter through current node
            left_height = calculate_height(node.left)
            right_height = calculate_height(node.right)
            diameter_through_node = left_height + right_height
            max_diameter = max(max_diameter, diameter_through_node)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return max_diameter
    
    def diameterOfBinaryTree_with_path(self, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """
        Approach 5: Find Diameter with Actual Path
        
        Return both diameter length and the actual path nodes.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return 0, []
        
        max_diameter = 0
        diameter_path = []
        
        def dfs_with_path(node):
            nonlocal max_diameter, diameter_path
            
            if not node:
                return 0, []
            
            left_height, left_path = dfs_with_path(node.left)
            right_height, right_path = dfs_with_path(node.right)
            
            # Current diameter through this node
            current_diameter = left_height + right_height
            
            if current_diameter > max_diameter:
                max_diameter = current_diameter
                # Construct diameter path
                diameter_path = left_path[::-1] + [node.val] + right_path
            
            # Return height and path for this subtree
            if left_height > right_height:
                return left_height + 1, left_path + [node.val]
            else:
                return right_height + 1, right_path + [node.val]
        
        dfs_with_path(root)
        return max_diameter, diameter_path
    
    def diameterOfBinaryTree_memoized(self, root: Optional[TreeNode]) -> int:
        """
        Approach 6: Memoized Height Calculation
        
        Cache height calculations to avoid recomputation.
        
        Time: O(n)
        Space: O(n) - memoization cache
        """
        if not root:
            return 0
        
        height_cache = {}
        
        def get_height(node):
            if not node:
                return 0
            
            if node in height_cache:
                return height_cache[node]
            
            left_height = get_height(node.left)
            right_height = get_height(node.right)
            height_cache[node] = 1 + max(left_height, right_height)
            
            return height_cache[node]
        
        max_diameter = 0
        
        def find_diameter(node):
            nonlocal max_diameter
            
            if not node:
                return
            
            left_height = get_height(node.left)
            right_height = get_height(node.right)
            diameter_through_node = left_height + right_height
            max_diameter = max(max_diameter, diameter_through_node)
            
            find_diameter(node.left)
            find_diameter(node.right)
        
        find_diameter(root)
        return max_diameter
    
    def diameterOfBinaryTree_morris_based(self, root: Optional[TreeNode]) -> int:
        """
        Approach 7: Morris Traversal Based (Space Optimized)
        
        Attempt to use Morris traversal for space optimization.
        Note: This is complex for diameter calculation.
        
        Time: O(n)
        Space: O(1) for traversal, O(n) for height storage
        """
        if not root:
            return 0
        
        # Since Morris traversal modifies tree structure temporarily,
        # we need to store heights during traversal
        node_heights = {}
        max_diameter = 0
        
        def calculate_height_morris(node):
            if not node:
                return 0
            
            current = node
            height = 0
            
            while current:
                if not current.left:
                    height += 1
                    current = current.right
                else:
                    predecessor = current.left
                    temp_height = 1
                    
                    while predecessor.right and predecessor.right != current:
                        predecessor = predecessor.right
                        temp_height += 1
                    
                    if not predecessor.right:
                        predecessor.right = current
                        height += 1
                        current = current.left
                    else:
                        predecessor.right = None
                        height = max(height, temp_height + 1)
                        current = current.right
            
            return height
        
        # For simplicity, use standard approach with Morris concept
        return self.diameterOfBinaryTree_recursive_optimized(root)
    
    def diameterOfBinaryTree_parallel_concept(self, root: Optional[TreeNode]) -> int:
        """
        Approach 8: Parallel Processing Concept
        
        Conceptual approach for parallel diameter calculation.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return 0
        
        # In a parallel setting, we could process left and right subtrees
        # simultaneously. Here we simulate the concept.
        
        results = {}
        
        def process_subtree(node, subtree_id):
            if not node:
                results[subtree_id] = (0, 0)  # (height, diameter)
                return
            
            # Process left and right (could be parallel)
            left_id = f"{subtree_id}_L"
            right_id = f"{subtree_id}_R"
            
            process_subtree(node.left, left_id)
            process_subtree(node.right, right_id)
            
            left_height, left_diameter = results.get(left_id, (0, 0))
            right_height, right_diameter = results.get(right_id, (0, 0))
            
            # Calculate for current node
            current_height = 1 + max(left_height, right_height)
            diameter_through_node = left_height + right_height
            current_diameter = max(diameter_through_node, left_diameter, right_diameter)
            
            results[subtree_id] = (current_height, current_diameter)
        
        process_subtree(root, "root")
        return results["root"][1]

def create_test_trees():
    """Create test trees for diameter calculation"""
    # Tree 1: [1,2,3,4,5] - diameter 3 (4->2->1->3 or 5->2->1->3)
    tree1 = TreeNode(1)
    tree1.left = TreeNode(2)
    tree1.right = TreeNode(3)
    tree1.left.left = TreeNode(4)
    tree1.left.right = TreeNode(5)
    
    # Tree 2: [1,2] - diameter 1
    tree2 = TreeNode(1)
    tree2.left = TreeNode(2)
    
    # Tree 3: Single node - diameter 0
    tree3 = TreeNode(1)
    
    # Tree 4: Linear tree - diameter 3
    tree4 = TreeNode(1)
    tree4.left = TreeNode(2)
    tree4.left.left = TreeNode(3)
    tree4.left.left.left = TreeNode(4)
    
    # Tree 5: Complex tree - diameter 6
    tree5 = TreeNode(1)
    tree5.left = TreeNode(2)
    tree5.right = TreeNode(3)
    tree5.left.left = TreeNode(4)
    tree5.left.right = TreeNode(5)
    tree5.left.left.left = TreeNode(8)
    tree5.left.left.right = TreeNode(9)
    tree5.left.right.left = TreeNode(10)
    tree5.left.right.right = TreeNode(11)
    
    return [
        (tree1, 3, "Balanced tree"),
        (tree2, 1, "Simple tree"),
        (tree3, 0, "Single node"),
        (tree4, 3, "Linear tree"),
        (tree5, 6, "Complex tree")
    ]

def test_diameter_algorithms():
    """Test all diameter algorithms"""
    test_trees = create_test_trees()
    
    print("=== Testing Diameter Algorithms ===")
    
    for tree, expected, description in test_trees:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        solver = DiameterBinaryTree()
        
        algorithms = [
            ("Recursive Optimized", solver.diameterOfBinaryTree_recursive_optimized),
            ("Two Pass", solver.diameterOfBinaryTree_two_pass),
            ("Iterative Postorder", solver.diameterOfBinaryTree_iterative_postorder),
            ("BFS Heights", solver.diameterOfBinaryTree_bfs_with_heights),
            ("Memoized", solver.diameterOfBinaryTree_memoized),
            ("Parallel Concept", solver.diameterOfBinaryTree_parallel_concept),
        ]
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(tree)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Diameter: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")
    
    # Test path finding
    print(f"\n--- Diameter Path Demo ---")
    tree, expected, description = test_trees[0]  # Use balanced tree
    solver = DiameterBinaryTree()
    diameter, path = solver.diameterOfBinaryTree_with_path(tree)
    print(f"Diameter: {diameter}, Path: {path}")

def demonstrate_diameter_concepts():
    """Demonstrate diameter concepts and properties"""
    print("\n=== Diameter Concepts ===")
    
    print("Key Properties:")
    print("• Diameter = longest path between any two nodes")
    print("• Path may or may not pass through root")
    print("• Length measured in number of edges")
    print("• For n nodes, diameter ranges from 0 to n-1")
    
    print("\nDiameter Calculation:")
    print("• For each node, consider path through that node")
    print("• Path length = left_height + right_height")
    print("• Global diameter = maximum over all nodes")
    print("• Can be calculated in single tree traversal")
    
    print("\nSpecial Cases:")
    print("• Empty tree: diameter = 0")
    print("• Single node: diameter = 0")
    print("• Linear tree: diameter = n-1")
    print("• Complete binary tree: diameter ≈ 2*log(n)")
    
    print("\nApplications:")
    print("• Network analysis (longest communication path)")
    print("• Tree structure analysis")
    print("• Algorithm complexity bounds")
    print("• Graph theory problems")

def analyze_diameter_complexity():
    """Analyze complexity of different diameter approaches"""
    print("\n=== Complexity Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Recursive Optimized (Recommended):**")
    print("   • Time: O(n) - single traversal")
    print("   • Space: O(h) - recursion stack")
    print("   • Pros: Optimal time, simple implementation")
    print("   • Cons: Recursion depth limited")
    
    print("\n2. **Two Pass:**")
    print("   • Time: O(n²) - height calculation for each node")
    print("   • Space: O(h)")
    print("   • Pros: Intuitive approach")
    print("   • Cons: Inefficient time complexity")
    
    print("\n3. **Iterative Postorder:**")
    print("   • Time: O(n)")
    print("   • Space: O(h)")
    print("   • Pros: No recursion limits")
    print("   • Cons: More complex implementation")
    
    print("\n4. **BFS with Heights:**")
    print("   • Time: O(n²) - BFS for each node")
    print("   • Space: O(w) - queue width")
    print("   • Pros: Level-by-level processing")
    print("   • Cons: Inefficient for diameter")
    
    print("\n5. **Memoized:**")
    print("   • Time: O(n)")
    print("   • Space: O(n) - memoization cache")
    print("   • Pros: Avoids recomputation")
    print("   • Cons: Extra space for cache")
    
    print("\nRecommendation: Use Recursive Optimized for best balance of simplicity and efficiency")

def demonstrate_diameter_variations():
    """Demonstrate variations of diameter problems"""
    print("\n=== Diameter Problem Variations ===")
    
    print("Related Problems:")
    
    print("\n1. **Tree Center:**")
    print("   • Node(s) that minimize maximum distance to any other node")
    print("   • Related to diameter endpoints")
    print("   • Used in tree rooting problems")
    
    print("\n2. **Tree Radius:**")
    print("   • Minimum eccentricity among all nodes")
    print("   • Radius = ⌈diameter/2⌉")
    print("   • Important for network design")
    
    print("\n3. **Weighted Diameter:**")
    print("   • Edges have weights, find maximum weighted path")
    print("   • Similar algorithm with weight accumulation")
    print("   • Applications in network latency analysis")
    
    print("\n4. **k-Diameter:**")
    print("   • Longest path using exactly k edges")
    print("   • More complex dynamic programming")
    print("   • Applications in bounded path problems")
    
    print("\n5. **Diameter in Other Structures:**")
    print("   • General graphs: All-pairs shortest paths")
    print("   • DAGs: Topological sort + DP")
    print("   • Weighted graphs: Modified algorithms")

if __name__ == "__main__":
    test_diameter_algorithms()
    demonstrate_diameter_concepts()
    analyze_diameter_complexity()
    demonstrate_diameter_variations()

"""
Diameter of Binary Tree - Key Insights:

1. **Problem Definition:**
   - Longest path between any two nodes in tree
   - Path may or may not pass through root
   - Length measured in number of edges
   - Fundamental tree property

2. **Algorithm Strategies:**
   - Single-pass: Calculate height and diameter together
   - Two-pass: Separate height and diameter calculations
   - Iterative: Avoid recursion using explicit stack
   - Memoized: Cache heights to avoid recomputation

3. **Key Insight:**
   - For each node, diameter through that node = left_height + right_height
   - Global diameter = maximum over all such local diameters
   - Can be computed efficiently in single traversal

4. **Complexity Considerations:**
   - Optimal: O(n) time, O(h) space
   - Naive: O(n²) time with repeated height calculations
   - Space-time tradeoffs in different approaches
   - Recursion vs iteration considerations

5. **Applications:**
   - Network analysis and design
   - Tree structure characterization
   - Algorithm complexity analysis
   - Graph theory and optimization

The diameter problem showcases efficient tree algorithms
and the power of combining multiple computations in single traversal.
"""
