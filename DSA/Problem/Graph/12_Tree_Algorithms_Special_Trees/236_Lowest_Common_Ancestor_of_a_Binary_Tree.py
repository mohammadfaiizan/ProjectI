"""
236. Lowest Common Ancestor of a Binary Tree - Multiple Approaches
Difficulty: Medium

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: "The lowest common ancestor is 
defined between two nodes p and q as the lowest node in T that has both p and q 
as descendants (where we allow a node to be a descendant of itself)."
"""

from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict, deque

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class LowestCommonAncestor:
    """Multiple approaches to find LCA in binary tree"""
    
    def lowestCommonAncestor_recursive(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Approach 1: Recursive Bottom-Up
        
        Time: O(n), Space: O(h)
        """
        if not root or root == p or root == q:
            return root
        
        left = self.lowestCommonAncestor_recursive(root.left, p, q)
        right = self.lowestCommonAncestor_recursive(root.right, p, q)
        
        if left and right:
            return root
        
        return left if left else right
    
    def lowestCommonAncestor_parent_pointers(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Approach 2: Parent Pointers
        
        Time: O(n), Space: O(n)
        """
        # Build parent map
        parent = {root: None}
        stack = [root]
        
        while p not in parent or q not in parent:
            node = stack.pop()
            
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
        
        # Get ancestors of p
        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]
        
        # Find first common ancestor of q
        while q not in ancestors:
            q = parent[q]
        
        return q
    
    def lowestCommonAncestor_path_comparison(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Approach 3: Path Comparison
        
        Time: O(n), Space: O(h)
        """
        def find_path(node, target, path):
            if not node:
                return False
            
            path.append(node)
            
            if node == target:
                return True
            
            if (find_path(node.left, target, path) or 
                find_path(node.right, target, path)):
                return True
            
            path.pop()
            return False
        
        path_p, path_q = [], []
        find_path(root, p, path_p)
        find_path(root, q, path_q)
        
        # Find last common node in paths
        lca = None
        for i in range(min(len(path_p), len(path_q))):
            if path_p[i] == path_q[i]:
                lca = path_p[i]
            else:
                break
        
        return lca
    
    def lowestCommonAncestor_iterative_postorder(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Approach 4: Iterative Postorder
        
        Time: O(n), Space: O(h)
        """
        stack = []
        current = root
        last_visited = None
        found = {p: False, q: False}
        lca_candidate = None
        
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
                    
                    # Check if this node is p or q
                    if node == p or node == q:
                        found[node] = True
                        if not lca_candidate:
                            lca_candidate = node
                    
                    # Check if both p and q found in subtree
                    if found[p] and found[q] and not lca_candidate:
                        lca_candidate = node
                    
                    last_visited = node
        
        return lca_candidate
    
    def lowestCommonAncestor_euler_tour(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Approach 5: Euler Tour + RMQ
        
        Time: O(n), Space: O(n)
        """
        # Euler tour
        tour = []
        first_occurrence = {}
        depth = []
        
        def euler_dfs(node, d):
            if not node:
                return
            
            if node not in first_occurrence:
                first_occurrence[node] = len(tour)
            
            tour.append(node)
            depth.append(d)
            
            if node.left:
                euler_dfs(node.left, d + 1)
                tour.append(node)
                depth.append(d)
            
            if node.right:
                euler_dfs(node.right, d + 1)
                tour.append(node)
                depth.append(d)
        
        euler_dfs(root, 0)
        
        # Find LCA using RMQ on depth array
        left_idx = first_occurrence[p]
        right_idx = first_occurrence[q]
        
        if left_idx > right_idx:
            left_idx, right_idx = right_idx, left_idx
        
        min_depth_idx = left_idx
        for i in range(left_idx, right_idx + 1):
            if depth[i] < depth[min_depth_idx]:
                min_depth_idx = i
        
        return tour[min_depth_idx]

def create_test_tree():
    """Create test tree for LCA testing"""
    #       3
    #      / \
    #     5   1
    #    / \ / \
    #   6  2 0  8
    #     / \
    #    7   4
    
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)
    root.right.left = TreeNode(0)
    root.right.right = TreeNode(8)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(4)
    
    return root

def test_lca_algorithms():
    """Test all LCA algorithms"""
    solver = LowestCommonAncestor()
    root = create_test_tree()
    
    # Test cases: (p_val, q_val, expected_lca_val)
    test_cases = [
        (5, 1, 3),  # LCA of 5 and 1 is 3
        (5, 4, 5),  # LCA of 5 and 4 is 5
        (6, 2, 5),  # LCA of 6 and 2 is 5
        (7, 4, 2),  # LCA of 7 and 4 is 2
    ]
    
    # Create node references
    nodes = {}
    def collect_nodes(node):
        if node:
            nodes[node.val] = node
            collect_nodes(node.left)
            collect_nodes(node.right)
    
    collect_nodes(root)
    
    algorithms = [
        ("Recursive", solver.lowestCommonAncestor_recursive),
        ("Parent Pointers", solver.lowestCommonAncestor_parent_pointers),
        ("Path Comparison", solver.lowestCommonAncestor_path_comparison),
        ("Euler Tour", solver.lowestCommonAncestor_euler_tour),
    ]
    
    print("=== Testing LCA Algorithms ===")
    
    for p_val, q_val, expected in test_cases:
        print(f"\n--- LCA({p_val}, {q_val}) = {expected} ---")
        
        p_node = nodes[p_val]
        q_node = nodes[q_val]
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(root, p_node, q_node)
                result_val = result.val if result else None
                status = "✓" if result_val == expected else "✗"
                print(f"{alg_name:15} | {status} | LCA: {result_val}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_lca_algorithms()

"""
LCA algorithms demonstrate various tree traversal and preprocessing techniques
for efficient ancestor queries in binary trees.
"""
