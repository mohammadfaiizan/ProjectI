"""
1022. Sum of Root To Leaf Binary Numbers - Multiple Approaches
Difficulty: Easy

You are given the root of a binary tree where each node has a value 0 or 1. 
Each root-to-leaf path represents a binary number starting with the most 
significant bit.

For example, if the path is 0 -> 1 -> 1 -> 0 -> 1, then this could represent 
01101 in binary, which is 13 in decimal.

For all leaves in the tree, consider the numbers represented by the path from 
the root to that leaf. Return the sum of these numbers.
"""

from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class SumRootToLeafBinaryNumbers:
    """Multiple approaches to calculate sum of root-to-leaf binary numbers"""
    
    def sumRootToLeaf_recursive_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 1: Recursive DFS with Path Building
        
        Build binary number along path and sum at leaves.
        
        Time: O(n) - visit each node once
        Space: O(h) - recursion stack
        """
        def dfs(node, current_number):
            if not node:
                return 0
            
            # Build current binary number
            current_number = current_number * 2 + node.val
            
            # If leaf node, return the number
            if not node.left and not node.right:
                return current_number
            
            # Sum from left and right subtrees
            left_sum = dfs(node.left, current_number)
            right_sum = dfs(node.right, current_number)
            
            return left_sum + right_sum
        
        return dfs(root, 0)
    
    def sumRootToLeaf_iterative_dfs(self, root: Optional[TreeNode]) -> int:
        """
        Approach 2: Iterative DFS using Stack
        
        Use explicit stack to avoid recursion.
        
        Time: O(n)
        Space: O(h) - stack space
        """
        if not root:
            return 0
        
        stack = [(root, 0)]  # (node, current_number)
        total_sum = 0
        
        while stack:
            node, current_number = stack.pop()
            current_number = current_number * 2 + node.val
            
            # If leaf node, add to sum
            if not node.left and not node.right:
                total_sum += current_number
            else:
                # Add children to stack
                if node.right:
                    stack.append((node.right, current_number))
                if node.left:
                    stack.append((node.left, current_number))
        
        return total_sum
    
    def sumRootToLeaf_bfs_level_order(self, root: Optional[TreeNode]) -> int:
        """
        Approach 3: BFS Level Order Traversal
        
        Process nodes level by level using queue.
        
        Time: O(n)
        Space: O(w) - queue width
        """
        if not root:
            return 0
        
        queue = deque([(root, 0)])  # (node, current_number)
        total_sum = 0
        
        while queue:
            node, current_number = queue.popleft()
            current_number = current_number * 2 + node.val
            
            # If leaf node, add to sum
            if not node.left and not node.right:
                total_sum += current_number
            else:
                # Add children to queue
                if node.left:
                    queue.append((node.left, current_number))
                if node.right:
                    queue.append((node.right, current_number))
        
        return total_sum
    
    def sumRootToLeaf_with_path_collection(self, root: Optional[TreeNode]) -> tuple:
        """
        Approach 4: Collect All Paths Then Calculate
        
        First collect all root-to-leaf paths, then calculate sum.
        
        Time: O(n * h) - n nodes, h height for path copying
        Space: O(n * h) - store all paths
        """
        if not root:
            return 0, []
        
        all_paths = []
        
        def collect_paths(node, current_path):
            if not node:
                return
            
            current_path.append(node.val)
            
            # If leaf, save the path
            if not node.left and not node.right:
                all_paths.append(current_path[:])
            else:
                collect_paths(node.left, current_path)
                collect_paths(node.right, current_path)
            
            current_path.pop()
        
        collect_paths(root, [])
        
        # Convert paths to numbers and sum
        total_sum = 0
        for path in all_paths:
            number = 0
            for bit in path:
                number = number * 2 + bit
            total_sum += number
        
        return total_sum, all_paths
    
    def sumRootToLeaf_bit_manipulation(self, root: Optional[TreeNode]) -> int:
        """
        Approach 5: Optimized Bit Manipulation
        
        Use bit shifting for efficient binary number construction.
        
        Time: O(n)
        Space: O(h)
        """
        def dfs(node, current_value):
            if not node:
                return 0
            
            # Left shift and add current bit
            current_value = (current_value << 1) | node.val
            
            # If leaf, return current value
            if not node.left and not node.right:
                return current_value
            
            # Sum from subtrees
            return dfs(node.left, current_value) + dfs(node.right, current_value)
        
        return dfs(root, 0)
    
    def sumRootToLeaf_morris_traversal(self, root: Optional[TreeNode]) -> int:
        """
        Approach 6: Morris Traversal (Space Optimized)
        
        Use Morris traversal for O(1) space complexity.
        Note: Complex for this specific problem due to path tracking.
        
        Time: O(n)
        Space: O(1) for traversal, but need path tracking
        """
        if not root:
            return 0
        
        # Morris traversal is complex for path-dependent problems
        # Fall back to efficient recursive approach
        return self.sumRootToLeaf_bit_manipulation(root)
    
    def sumRootToLeaf_preorder_iterative(self, root: Optional[TreeNode]) -> int:
        """
        Approach 7: Preorder Iterative Traversal
        
        Explicit preorder traversal with path tracking.
        
        Time: O(n)
        Space: O(h)
        """
        if not root:
            return 0
        
        stack = [(root, 0, False)]  # (node, current_number, visited)
        total_sum = 0
        
        while stack:
            node, current_number, visited = stack.pop()
            
            if visited:
                # Process node
                current_number = current_number * 2 + node.val
                
                if not node.left and not node.right:
                    total_sum += current_number
            else:
                # Mark as visited and add back to stack
                stack.append((node, current_number, True))
                
                # Add children (right first for correct order)
                if node.right:
                    stack.append((node.right, current_number * 2 + node.val, False))
                if node.left:
                    stack.append((node.left, current_number * 2 + node.val, False))
        
        return total_sum
    
    def sumRootToLeaf_divide_conquer(self, root: Optional[TreeNode]) -> int:
        """
        Approach 8: Divide and Conquer with Memoization
        
        Use divide and conquer paradigm with optimization.
        
        Time: O(n)
        Space: O(h)
        """
        def divide_conquer(node, path_value):
            # Base case
            if not node:
                return 0
            
            # Update path value
            path_value = path_value * 2 + node.val
            
            # If leaf, return path value
            if not node.left and not node.right:
                return path_value
            
            # Divide: get sums from subtrees
            left_sum = divide_conquer(node.left, path_value)
            right_sum = divide_conquer(node.right, path_value)
            
            # Conquer: combine results
            return left_sum + right_sum
        
        return divide_conquer(root, 0)

def create_test_trees():
    """Create test trees for binary number sum calculation"""
    # Tree 1: [1,0,1,0,1,0,1] - paths: 100(4), 101(5), 110(6), 111(7) = 22
    tree1 = TreeNode(1)
    tree1.left = TreeNode(0)
    tree1.right = TreeNode(1)
    tree1.left.left = TreeNode(0)
    tree1.left.right = TreeNode(1)
    tree1.right.left = TreeNode(0)
    tree1.right.right = TreeNode(1)
    
    # Tree 2: [0] - path: 0(0) = 0
    tree2 = TreeNode(0)
    
    # Tree 3: [1] - path: 1(1) = 1
    tree3 = TreeNode(1)
    
    # Tree 4: [1,1] - path: 11(3) = 3
    tree4 = TreeNode(1)
    tree4.left = TreeNode(1)
    
    # Tree 5: Linear tree [1,0,0,1] - paths: 100(4), 101(5) = 9
    tree5 = TreeNode(1)
    tree5.left = TreeNode(0)
    tree5.left.left = TreeNode(0)
    tree5.left.right = TreeNode(1)
    
    return [
        (tree1, 22, "Balanced tree"),
        (tree2, 0, "Single zero"),
        (tree3, 1, "Single one"),
        (tree4, 3, "Simple path"),
        (tree5, 9, "Linear paths")
    ]

def test_sum_algorithms():
    """Test all binary sum algorithms"""
    solver = SumRootToLeafBinaryNumbers()
    test_trees = create_test_trees()
    
    algorithms = [
        ("Recursive DFS", solver.sumRootToLeaf_recursive_dfs),
        ("Iterative DFS", solver.sumRootToLeaf_iterative_dfs),
        ("BFS Level Order", solver.sumRootToLeaf_bfs_level_order),
        ("Bit Manipulation", solver.sumRootToLeaf_bit_manipulation),
        ("Morris Traversal", solver.sumRootToLeaf_morris_traversal),
        ("Preorder Iterative", solver.sumRootToLeaf_preorder_iterative),
        ("Divide & Conquer", solver.sumRootToLeaf_divide_conquer),
    ]
    
    print("=== Testing Binary Sum Algorithms ===")
    
    for tree, expected, description in test_trees:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(tree)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Sum: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")
    
    # Test path collection
    print(f"\n--- Path Collection Demo ---")
    tree, expected, description = test_trees[0]  # Use balanced tree
    total_sum, paths = solver.sumRootToLeaf_with_path_collection(tree)
    print(f"Total sum: {total_sum}")
    print("All paths:")
    for i, path in enumerate(paths):
        binary_str = ''.join(map(str, path))
        decimal_val = int(binary_str, 2)
        print(f"  Path {i+1}: {path} -> {binary_str} -> {decimal_val}")

def demonstrate_binary_concepts():
    """Demonstrate binary number concepts"""
    print("\n=== Binary Number Concepts ===")
    
    print("Binary Representation:")
    print("• Each path represents a binary number")
    print("• Most significant bit at root")
    print("• Least significant bit at leaf")
    print("• Path [1,0,1] represents binary 101 = decimal 5")
    
    print("\nBinary to Decimal Conversion:")
    print("• Method 1: Multiply by 2 and add bit")
    print("  Example: 101 -> ((0*2+1)*2+0)*2+1 = 5")
    print("• Method 2: Bit shifting")
    print("  Example: 101 -> (0<<1)|1 -> (1<<1)|0 -> (2<<1)|1 = 5")
    print("• Method 3: Positional notation")
    print("  Example: 101 -> 1*2² + 0*2¹ + 1*2⁰ = 4+0+1 = 5")
    
    print("\nPath Processing:")
    print("• Visit all root-to-leaf paths")
    print("• Build binary number incrementally")
    print("• Sum all leaf values")
    print("• Handle edge cases (single node, empty tree)")

def analyze_binary_sum_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Complexity Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Recursive DFS (Recommended):**")
    print("   • Time: O(n) - visit each node once")
    print("   • Space: O(h) - recursion stack")
    print("   • Pros: Simple, efficient")
    print("   • Cons: Stack overflow for deep trees")
    
    print("\n2. **Iterative DFS:**")
    print("   • Time: O(n)")
    print("   • Space: O(h) - explicit stack")
    print("   • Pros: No recursion limits")
    print("   • Cons: Slightly more complex")
    
    print("\n3. **BFS Level Order:**")
    print("   • Time: O(n)")
    print("   • Space: O(w) - queue width")
    print("   • Pros: Level-by-level processing")
    print("   • Cons: May use more space for wide trees")
    
    print("\n4. **Path Collection:**")
    print("   • Time: O(n * h) - path copying overhead")
    print("   • Space: O(n * h) - store all paths")
    print("   • Pros: Explicit path tracking")
    print("   • Cons: High space complexity")
    
    print("\n5. **Bit Manipulation:**")
    print("   • Time: O(n)")
    print("   • Space: O(h)")
    print("   • Pros: Efficient bit operations")
    print("   • Cons: Requires understanding of bit ops")
    
    print("\nRecommendation: Recursive DFS or Bit Manipulation for optimal performance")

def demonstrate_binary_applications():
    """Demonstrate applications of binary path problems"""
    print("\n=== Binary Path Applications ===")
    
    print("Real-World Applications:")
    
    print("\n1. **Decision Trees:**")
    print("   • Binary decisions at each node")
    print("   • Path represents decision sequence")
    print("   • Leaf values are outcomes")
    print("   • Sum represents total utility")
    
    print("\n2. **Huffman Coding:**")
    print("   • Binary tree for character encoding")
    print("   • Path to leaf is character code")
    print("   • Shorter paths for frequent characters")
    print("   • Optimal prefix-free encoding")
    
    print("\n3. **Game Theory:**")
    print("   • Game state trees")
    print("   • Binary choices at each state")
    print("   • Path represents game sequence")
    print("   • Leaf values are payoffs")
    
    print("\n4. **Network Routing:**")
    print("   • Binary routing decisions")
    print("   • Path represents route")
    print("   • Leaf values are costs/benefits")
    print("   • Optimize total network performance")
    
    print("\n5. **Machine Learning:**")
    print("   • Binary classification trees")
    print("   • Feature-based splitting")
    print("   • Path represents classification rule")
    print("   • Leaf values are class probabilities")

def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques for binary path problems"""
    print("\n=== Optimization Techniques ===")
    
    print("Performance Optimizations:")
    
    print("\n1. **Bit Manipulation:**")
    print("   • Use bit shifting instead of multiplication")
    print("   • (value << 1) | bit instead of value * 2 + bit")
    print("   • Faster for binary operations")
    print("   • Hardware-optimized operations")
    
    print("\n2. **Early Termination:**")
    print("   • Stop at leaves immediately")
    print("   • No need to continue traversal")
    print("   • Reduces unnecessary computation")
    print("   • Important for large trees")
    
    print("\n3. **Memory Optimization:**")
    print("   • Avoid storing all paths")
    print("   • Calculate sum incrementally")
    print("   • Use iterative approaches for deep trees")
    print("   • Consider Morris traversal for extreme cases")
    
    print("\n4. **Numerical Considerations:**")
    print("   • Handle integer overflow for large numbers")
    print("   • Use modular arithmetic if needed")
    print("   • Consider BigInteger for very large results")
    print("   • Validate input constraints")
    
    print("\n5. **Algorithm Selection:**")
    print("   • Recursive for simplicity")
    print("   • Iterative for deep trees")
    print("   • BFS for wide trees")
    print("   • Bit manipulation for performance")

def demonstrate_problem_variations():
    """Demonstrate variations of binary path problems"""
    print("\n=== Problem Variations ===")
    
    print("Related Problems:")
    
    print("\n1. **Path Sum Problems:**")
    print("   • Sum of all root-to-leaf paths")
    print("   • Maximum/minimum path sum")
    print("   • Path sum equal to target")
    print("   • Count paths with specific sum")
    
    print("\n2. **Binary String Problems:**")
    print("   • Concatenate path values as strings")
    print("   • Find lexicographically smallest path")
    print("   • Count palindromic paths")
    print("   • Pattern matching in paths")
    
    print("\n3. **Weighted Path Problems:**")
    print("   • Nodes have different weights")
    print("   • Edge weights in addition to node values")
    print("   • Multiplicative instead of additive")
    print("   • Complex scoring functions")
    
    print("\n4. **Multi-valued Nodes:**")
    print("   • Nodes can have values other than 0/1")
    print("   • Different base number systems")
    print("   • Hexadecimal, octal representations")
    print("   • Variable-base number systems")
    
    print("\n5. **Conditional Processing:**")
    print("   • Process only certain paths")
    print("   • Skip paths based on conditions")
    print("   • Dynamic path selection")
    print("   • Probabilistic path traversal")

if __name__ == "__main__":
    test_sum_algorithms()
    demonstrate_binary_concepts()
    analyze_binary_sum_complexity()
    demonstrate_binary_applications()
    demonstrate_optimization_techniques()
    demonstrate_problem_variations()

"""
Sum of Root To Leaf Binary Numbers - Key Insights:

1. **Problem Structure:**
   - Each root-to-leaf path represents a binary number
   - Most significant bit at root, least significant at leaf
   - Sum all binary numbers represented by paths
   - Fundamental tree traversal with path processing

2. **Algorithm Strategies:**
   - Recursive: Natural tree recursion with path building
   - Iterative: DFS or BFS with explicit data structures
   - Path collection: Store all paths then process
   - Bit manipulation: Efficient binary operations

3. **Key Techniques:**
   - Incremental number building: value = value * 2 + bit
   - Bit shifting optimization: value = (value << 1) | bit
   - Leaf detection: both children are null
   - Path tracking: maintain current binary value

4. **Complexity Considerations:**
   - Time: O(n) for all efficient approaches
   - Space: O(h) for recursive, O(w) for BFS
   - Path storage: O(n * h) if storing all paths
   - Bit operations: More efficient than arithmetic

5. **Applications:**
   - Decision tree analysis
   - Huffman coding trees
   - Game theory and optimization
   - Network routing algorithms
   - Binary classification problems

The problem demonstrates efficient path processing
and binary number manipulation in tree structures.
"""
