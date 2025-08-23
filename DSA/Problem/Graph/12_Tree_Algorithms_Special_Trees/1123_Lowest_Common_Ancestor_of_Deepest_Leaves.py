"""
1123. Lowest Common Ancestor of Deepest Leaves - Multiple Approaches
Difficulty: Medium

Given the root of a binary tree, return the lowest common ancestor of its deepest leaves.

Recall that:
- The node of a binary tree is a leaf if and only if it has no children
- The depth of the root of the tree is 0. If the depth of a node is d, the depth of each of its children is d + 1.
- The lowest common ancestor of a set S of nodes, is the node A with the largest depth such that every node in S is in the subtree with root A.
"""

from typing import Optional, Tuple

# Definition for a binary tree node
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class LowestCommonAncestorDeepestLeaves:
    """Multiple approaches to find LCA of deepest leaves"""
    
    def lcaDeepestLeaves_two_pass(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 1: Two-Pass Algorithm
        
        First pass: find maximum depth
        Second pass: find LCA of nodes at maximum depth
        
        Time: O(N), Space: O(H)
        """
        if not root:
            return None
        
        # First pass: find maximum depth
        def find_max_depth(node: TreeNode) -> int:
            if not node:
                return 0
            return 1 + max(find_max_depth(node.left), find_max_depth(node.right))
        
        max_depth = find_max_depth(root)
        
        # Second pass: find LCA of deepest leaves
        def find_lca(node: TreeNode, depth: int) -> Optional[TreeNode]:
            if not node:
                return None
            
            if depth == max_depth:
                return node  # This is a deepest leaf
            
            left_lca = find_lca(node.left, depth + 1)
            right_lca = find_lca(node.right, depth + 1)
            
            # If both subtrees have deepest leaves, current node is LCA
            if left_lca and right_lca:
                return node
            
            # Return the subtree that contains deepest leaves
            return left_lca or right_lca
        
        return find_lca(root, 1)
    
    def lcaDeepestLeaves_one_pass(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 2: One-Pass Algorithm
        
        Single DFS that returns both depth and LCA information.
        
        Time: O(N), Space: O(H)
        """
        def dfs(node: TreeNode) -> Tuple[int, TreeNode]:
            """
            Returns (depth, lca_of_deepest_leaves_in_subtree)
            """
            if not node:
                return 0, None
            
            left_depth, left_lca = dfs(node.left)
            right_depth, right_lca = dfs(node.right)
            
            if left_depth > right_depth:
                # Deepest leaves are in left subtree
                return left_depth + 1, left_lca
            elif right_depth > left_depth:
                # Deepest leaves are in right subtree
                return right_depth + 1, right_lca
            else:
                # Both subtrees have same depth, current node is LCA
                return left_depth + 1, node
        
        _, lca = dfs(root)
        return lca
    
    def lcaDeepestLeaves_bottom_up(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 3: Bottom-Up Approach
        
        Build solution from leaves up to root.
        
        Time: O(N), Space: O(H)
        """
        def helper(node: TreeNode) -> Tuple[int, TreeNode]:
            """
            Returns (max_depth_in_subtree, lca_of_deepest_leaves)
            """
            if not node:
                return 0, None
            
            if not node.left and not node.right:
                # Leaf node
                return 1, node
            
            left_depth, left_lca = helper(node.left) if node.left else (0, None)
            right_depth, right_lca = helper(node.right) if node.right else (0, None)
            
            if left_depth > right_depth:
                return left_depth + 1, left_lca
            elif right_depth > left_depth:
                return right_depth + 1, right_lca
            else:
                # Equal depths - current node is LCA
                return left_depth + 1, node
        
        _, result = helper(root)
        return result
    
    def lcaDeepestLeaves_path_tracking(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 4: Path Tracking Approach
        
        Find all paths to deepest leaves, then find their LCA.
        
        Time: O(N), Space: O(N)
        """
        if not root:
            return None
        
        # Find all deepest leaves and their paths
        deepest_paths = []
        max_depth = 0
        
        def find_deepest_paths(node: TreeNode, path: list, depth: int):
            nonlocal max_depth
            
            if not node:
                return
            
            path.append(node)
            
            if not node.left and not node.right:
                # Leaf node
                if depth > max_depth:
                    max_depth = depth
                    deepest_paths.clear()
                    deepest_paths.append(path[:])
                elif depth == max_depth:
                    deepest_paths.append(path[:])
            
            find_deepest_paths(node.left, path, depth + 1)
            find_deepest_paths(node.right, path, depth + 1)
            
            path.pop()
        
        find_deepest_paths(root, [], 1)
        
        if not deepest_paths:
            return root
        
        # Find LCA of all deepest paths
        if len(deepest_paths) == 1:
            return deepest_paths[0][-1]  # Single deepest leaf
        
        # Find common prefix of all paths
        min_length = min(len(path) for path in deepest_paths)
        lca_index = 0
        
        for i in range(min_length):
            if all(path[i] == deepest_paths[0][i] for path in deepest_paths):
                lca_index = i
            else:
                break
        
        return deepest_paths[0][lca_index]
    
    def lcaDeepestLeaves_iterative(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 5: Iterative Approach using Stack
        
        Use iterative DFS with explicit stack.
        
        Time: O(N), Space: O(N)
        """
        if not root:
            return None
        
        # Stack stores (node, depth, parent)
        stack = [(root, 1, None)]
        node_info = {}  # node -> (depth, parent)
        max_depth = 0
        deepest_leaves = []
        
        # Find all nodes and their depths
        while stack:
            node, depth, parent = stack.pop()
            node_info[node] = (depth, parent)
            max_depth = max(max_depth, depth)
            
            if node.right:
                stack.append((node.right, depth + 1, node))
            if node.left:
                stack.append((node.left, depth + 1, node))
        
        # Find all deepest leaves
        for node, (depth, parent) in node_info.items():
            if depth == max_depth:
                deepest_leaves.append(node)
        
        # Find LCA of deepest leaves
        if len(deepest_leaves) == 1:
            return deepest_leaves[0]
        
        # Use set intersection to find common ancestors
        def get_ancestors(node):
            ancestors = set()
            current = node
            while current:
                ancestors.add(current)
                current = node_info[current][1] if current in node_info else None
            return ancestors
        
        # Find intersection of all ancestor sets
        common_ancestors = get_ancestors(deepest_leaves[0])
        for leaf in deepest_leaves[1:]:
            common_ancestors &= get_ancestors(leaf)
        
        # Find the deepest common ancestor
        lca = None
        max_ancestor_depth = -1
        
        for ancestor in common_ancestors:
            ancestor_depth = node_info[ancestor][0]
            if ancestor_depth > max_ancestor_depth:
                max_ancestor_depth = ancestor_depth
                lca = ancestor
        
        return lca
    
    def lcaDeepestLeaves_recursive_with_memo(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach 6: Recursive with Memoization
        
        Use memoization to cache depth calculations.
        
        Time: O(N), Space: O(N)
        """
        if not root:
            return None
        
        depth_memo = {}
        
        def get_depth(node: TreeNode) -> int:
            if not node:
                return 0
            
            if node in depth_memo:
                return depth_memo[node]
            
            depth = 1 + max(get_depth(node.left), get_depth(node.right))
            depth_memo[node] = depth
            return depth
        
        def find_lca(node: TreeNode) -> TreeNode:
            if not node:
                return None
            
            left_depth = get_depth(node.left)
            right_depth = get_depth(node.right)
            
            if left_depth > right_depth:
                return find_lca(node.left)
            elif right_depth > left_depth:
                return find_lca(node.right)
            else:
                return node
        
        return find_lca(root)

def test_lca_deepest_leaves():
    """Test LCA of deepest leaves algorithms"""
    solver = LowestCommonAncestorDeepestLeaves()
    
    # Create test tree:      3
    #                       / \
    #                      5   1
    #                     / \   \
    #                    6   2   0
    #                       / \
    #                      7   4
    
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)
    root.right.right = TreeNode(0)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(4)
    
    expected_val = 2  # Node 2 is LCA of deepest leaves (7, 4, 0)
    
    algorithms = [
        ("Two-Pass", solver.lcaDeepestLeaves_two_pass),
        ("One-Pass", solver.lcaDeepestLeaves_one_pass),
        ("Bottom-Up", solver.lcaDeepestLeaves_bottom_up),
        ("Path Tracking", solver.lcaDeepestLeaves_path_tracking),
        ("Iterative", solver.lcaDeepestLeaves_iterative),
        ("Recursive with Memo", solver.lcaDeepestLeaves_recursive_with_memo),
    ]
    
    print("=== Testing LCA of Deepest Leaves ===")
    print(f"Expected LCA value: {expected_val}")
    
    for alg_name, alg_func in algorithms:
        try:
            result = alg_func(root)
            result_val = result.val if result else None
            status = "✓" if result_val == expected_val else "✗"
            print(f"{alg_name:22} | {status} | LCA value: {result_val}")
        except Exception as e:
            print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_lca_deepest_leaves()

"""
Lowest Common Ancestor of Deepest Leaves demonstrates
advanced tree traversal techniques, depth analysis,
and lowest common ancestor algorithms in binary trees.
"""
