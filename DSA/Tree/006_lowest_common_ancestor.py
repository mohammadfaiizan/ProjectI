"""
Lowest Common Ancestor (LCA) Techniques - Advanced LCA Algorithms
This module implements various LCA algorithms from basic to advanced techniques.
"""

from collections import deque, defaultdict
from typing import List, Optional, Dict, Tuple
import math

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class LowestCommonAncestor:
    
    def __init__(self):
        """Initialize LCA algorithms"""
        pass
    
    # ==================== BASIC LCA IN BINARY TREE ====================
    
    def lca_binary_tree_recursive(self, root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
        """
        Find LCA in binary tree using recursive approach
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        
        Args:
            root: Root of the tree
            p, q: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        if not root or root == p or root == q:
            return root
        
        # Search in left and right subtrees
        left_lca = self.lca_binary_tree_recursive(root.left, p, q)
        right_lca = self.lca_binary_tree_recursive(root.right, p, q)
        
        # If both left and right return non-null, root is LCA
        if left_lca and right_lca:
            return root
        
        # Return non-null result
        return left_lca if left_lca else right_lca
    
    def lca_with_parent_pointers(self, p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
        """
        Find LCA when nodes have parent pointers
        
        Args:
            p, q: Two nodes with parent pointers
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        # Get all ancestors of p
        ancestors = set()
        current = p
        while current:
            ancestors.add(current)
            current = getattr(current, 'parent', None)
        
        # Find first common ancestor of q
        current = q
        while current:
            if current in ancestors:
                return current
            current = getattr(current, 'parent', None)
        
        return None
    
    def lca_using_paths(self, root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
        """
        Find LCA by finding paths from root to both nodes
        
        Args:
            root: Root of the tree
            p, q: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
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
        
        path_p = []
        path_q = []
        
        if not find_path(root, p, path_p) or not find_path(root, q, path_q):
            return None
        
        # Find LCA by comparing paths
        lca = None
        min_len = min(len(path_p), len(path_q))
        
        for i in range(min_len):
            if path_p[i] == path_q[i]:
                lca = path_p[i]
            else:
                break
        
        return lca
    
    # ==================== LCA IN BST ====================
    
    def lca_bst_recursive(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Find LCA in BST using BST property (recursive)
        
        Time Complexity: O(h) where h is height
        Space Complexity: O(h) for recursion
        
        Args:
            root: Root of BST
            p, q: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        if not root:
            return None
        
        # If both p and q are smaller than root, LCA is in left subtree
        if p.val < root.val and q.val < root.val:
            return self.lca_bst_recursive(root.left, p, q)
        
        # If both p and q are greater than root, LCA is in right subtree
        if p.val > root.val and q.val > root.val:
            return self.lca_bst_recursive(root.right, p, q)
        
        # If one is smaller and one is greater, root is LCA
        return root
    
    def lca_bst_iterative(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        """
        Find LCA in BST using iterative approach
        
        Time Complexity: O(h)
        Space Complexity: O(1)
        
        Args:
            root: Root of BST
            p, q: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        while root:
            if p.val < root.val and q.val < root.val:
                root = root.left
            elif p.val > root.val and q.val > root.val:
                root = root.right
            else:
                return root
        
        return None
    
    # ==================== BINARY LIFTING FOR LCA ====================
    
    def build_binary_lifting_table(self, root: Optional[TreeNode], n: int) -> Tuple[List[List[int]], List[int]]:
        """
        Build binary lifting table for fast LCA queries
        
        Time Complexity: O(n log n) for preprocessing
        Space Complexity: O(n log n)
        
        Args:
            root: Root of the tree
            n: Number of nodes
        
        Returns:
            Tuple of (parent table, depth array)
        """
        if not root:
            return [], []
        
        # Node mapping for binary lifting
        node_to_id = {}
        id_to_node = {}
        
        # Build node mapping using BFS
        queue = deque([root])
        node_id = 0
        node_to_id[root] = node_id
        id_to_node[node_id] = root
        node_id += 1
        
        while queue:
            node = queue.popleft()
            if node.left:
                node_to_id[node.left] = node_id
                id_to_node[node_id] = node.left
                queue.append(node.left)
                node_id += 1
            if node.right:
                node_to_id[node.right] = node_id
                id_to_node[node_id] = node.right
                queue.append(node.right)
                node_id += 1
        
        n = node_id
        LOG = int(math.log2(n)) + 1
        
        # parent[i][j] = 2^j-th ancestor of node i
        parent = [[-1] * LOG for _ in range(n)]
        depth = [0] * n
        
        # Build parent table and depth using DFS
        def dfs(node, par, d):
            node_id = node_to_id[node]
            depth[node_id] = d
            parent[node_id][0] = node_to_id[par] if par else -1
            
            # Fill binary lifting table
            for j in range(1, LOG):
                if parent[node_id][j-1] != -1:
                    parent[node_id][j] = parent[parent[node_id][j-1]][j-1]
            
            if node.left:
                dfs(node.left, node, d + 1)
            if node.right:
                dfs(node.right, node, d + 1)
        
        dfs(root, None, 0)
        
        self.parent_table = parent
        self.depth_table = depth
        self.node_to_id = node_to_id
        self.id_to_node = id_to_node
        self.LOG = LOG
        
        return parent, depth
    
    def lca_binary_lifting(self, u: TreeNode, v: TreeNode) -> Optional[TreeNode]:
        """
        Find LCA using binary lifting (after preprocessing)
        
        Time Complexity: O(log n) per query
        Space Complexity: O(1) per query
        
        Args:
            u, v: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        if not hasattr(self, 'parent_table'):
            return None
        
        u_id = self.node_to_id.get(u)
        v_id = self.node_to_id.get(v)
        
        if u_id is None or v_id is None:
            return None
        
        # Make sure u is deeper than v
        if self.depth_table[u_id] < self.depth_table[v_id]:
            u_id, v_id = v_id, u_id
        
        # Bring u to the same level as v
        diff = self.depth_table[u_id] - self.depth_table[v_id]
        
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u_id = self.parent_table[u_id][i]
        
        # If u and v are the same after leveling
        if u_id == v_id:
            return self.id_to_node[u_id]
        
        # Binary search for LCA
        for i in range(self.LOG - 1, -1, -1):
            if (self.parent_table[u_id][i] != self.parent_table[v_id][i]):
                u_id = self.parent_table[u_id][i]
                v_id = self.parent_table[v_id][i]
        
        return self.id_to_node[self.parent_table[u_id][0]]
    
    # ==================== EULER TOUR + RMQ FOR LCA ====================
    
    def build_euler_tour_rmq(self, root: Optional[TreeNode]) -> Tuple[List[int], List[int], Dict]:
        """
        Build Euler tour and RMQ for LCA queries
        
        Time Complexity: O(n log n) for preprocessing
        Space Complexity: O(n log n)
        
        Args:
            root: Root of the tree
        
        Returns:
            Tuple of (euler tour, depth array, first occurrence map)
        """
        if not root:
            return [], [], {}
        
        euler_tour = []
        depth_array = []
        first_occurrence = {}
        node_to_id = {}
        id_to_node = {}
        
        # Assign IDs to nodes
        def assign_ids(node, node_id):
            if not node:
                return node_id
            
            node_to_id[node] = node_id
            id_to_node[node_id] = node
            node_id += 1
            
            if node.left:
                node_id = assign_ids(node.left, node_id)
            if node.right:
                node_id = assign_ids(node.right, node_id)
            
            return node_id
        
        assign_ids(root, 0)
        
        # Build Euler tour
        def dfs_euler(node, depth):
            if not node:
                return
            
            node_id = node_to_id[node]
            
            # First occurrence
            if node_id not in first_occurrence:
                first_occurrence[node_id] = len(euler_tour)
            
            euler_tour.append(node_id)
            depth_array.append(depth)
            
            if node.left:
                dfs_euler(node.left, depth + 1)
                euler_tour.append(node_id)
                depth_array.append(depth)
            
            if node.right:
                dfs_euler(node.right, depth + 1)
                euler_tour.append(node_id)
                depth_array.append(depth)
        
        dfs_euler(root, 0)
        
        # Build sparse table for RMQ
        n = len(euler_tour)
        LOG = int(math.log2(n)) + 1
        st = [[0] * LOG for _ in range(n)]
        
        # Initialize sparse table
        for i in range(n):
            st[i][0] = i
        
        # Build sparse table
        for j in range(1, LOG):
            for i in range(n - (1 << j) + 1):
                left = st[i][j-1]
                right = st[i + (1 << (j-1))][j-1]
                
                if depth_array[left] < depth_array[right]:
                    st[i][j] = left
                else:
                    st[i][j] = right
        
        self.euler_tour = euler_tour
        self.depth_array = depth_array
        self.first_occurrence = first_occurrence
        self.sparse_table = st
        self.node_to_id_euler = node_to_id
        self.id_to_node_euler = id_to_node
        self.LOG_euler = LOG
        
        return euler_tour, depth_array, first_occurrence
    
    def lca_euler_rmq(self, u: TreeNode, v: TreeNode) -> Optional[TreeNode]:
        """
        Find LCA using Euler tour + RMQ
        
        Time Complexity: O(1) per query (after preprocessing)
        Space Complexity: O(1) per query
        
        Args:
            u, v: Two nodes to find LCA for
        
        Returns:
            TreeNode: Lowest common ancestor
        """
        if not hasattr(self, 'euler_tour'):
            return None
        
        u_id = self.node_to_id_euler.get(u)
        v_id = self.node_to_id_euler.get(v)
        
        if u_id is None or v_id is None:
            return None
        
        # Get first occurrences in Euler tour
        u_first = self.first_occurrence[u_id]
        v_first = self.first_occurrence[v_id]
        
        if u_first > v_first:
            u_first, v_first = v_first, u_first
        
        # RMQ query
        length = v_first - u_first + 1
        k = int(math.log2(length))
        
        left_min = self.sparse_table[u_first][k]
        right_min = self.sparse_table[v_first - (1 << k) + 1][k]
        
        if self.depth_array[left_min] < self.depth_array[right_min]:
            lca_tour_idx = left_min
        else:
            lca_tour_idx = right_min
        
        lca_id = self.euler_tour[lca_tour_idx]
        return self.id_to_node_euler[lca_id]
    
    # ==================== MULTIPLE LCA QUERIES ====================
    
    def lca_multiple_queries_naive(self, root: Optional[TreeNode], queries: List[Tuple[TreeNode, TreeNode]]) -> List[Optional[TreeNode]]:
        """
        Handle multiple LCA queries using naive approach
        
        Time Complexity: O(q * n) where q is number of queries
        
        Args:
            root: Root of the tree
            queries: List of (u, v) pairs
        
        Returns:
            List of LCA results
        """
        results = []
        for u, v in queries:
            lca = self.lca_binary_tree_recursive(root, u, v)
            results.append(lca)
        return results
    
    def lca_multiple_queries_optimized(self, root: Optional[TreeNode], queries: List[Tuple[TreeNode, TreeNode]]) -> List[Optional[TreeNode]]:
        """
        Handle multiple LCA queries using preprocessing
        
        Time Complexity: O(n log n + q) where q is number of queries
        
        Args:
            root: Root of the tree
            queries: List of (u, v) pairs
        
        Returns:
            List of LCA results
        """
        # Build Euler tour + RMQ
        self.build_euler_tour_rmq(root)
        
        results = []
        for u, v in queries:
            lca = self.lca_euler_rmq(u, v)
            results.append(lca)
        
        return results
    
    # ==================== SPECIAL LCA PROBLEMS ====================
    
    def lca_of_deepest_leaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Find LCA of deepest leaves in the tree
        
        Args:
            root: Root of the tree
        
        Returns:
            TreeNode: LCA of deepest leaves
        """
        def dfs(node):
            if not node:
                return 0, None
            
            left_depth, left_lca = dfs(node.left)
            right_depth, right_lca = dfs(node.right)
            
            if left_depth == right_depth:
                return left_depth + 1, node
            elif left_depth > right_depth:
                return left_depth + 1, left_lca
            else:
                return right_depth + 1, right_lca
        
        _, lca = dfs(root)
        return lca
    
    def distance_between_nodes(self, root: Optional[TreeNode], p: TreeNode, q: TreeNode) -> int:
        """
        Find distance between two nodes in the tree
        
        Args:
            root: Root of the tree
            p, q: Two nodes
        
        Returns:
            int: Distance between nodes
        """
        lca = self.lca_binary_tree_recursive(root, p, q)
        
        if not lca:
            return -1
        
        def get_distance(node, target, distance):
            if not node:
                return -1
            
            if node == target:
                return distance
            
            left_dist = get_distance(node.left, target, distance + 1)
            if left_dist != -1:
                return left_dist
            
            return get_distance(node.right, target, distance + 1)
        
        dist_p = get_distance(lca, p, 0)
        dist_q = get_distance(lca, q, 0)
        
        return dist_p + dist_q
    
    def kth_ancestor(self, node: TreeNode, k: int) -> Optional[TreeNode]:
        """
        Find kth ancestor of a node using binary lifting
        
        Args:
            node: Starting node
            k: Ancestor level (1 = parent, 2 = grandparent, etc.)
        
        Returns:
            TreeNode: kth ancestor or None if doesn't exist
        """
        if not hasattr(self, 'parent_table'):
            return None
        
        node_id = self.node_to_id.get(node)
        if node_id is None:
            return None
        
        # Check if kth ancestor exists
        if self.depth_table[node_id] < k:
            return None
        
        # Use binary lifting to find kth ancestor
        for i in range(self.LOG):
            if (k >> i) & 1:
                node_id = self.parent_table[node_id][i]
                if node_id == -1:
                    return None
        
        return self.id_to_node[node_id]
    
    # ==================== UTILITY METHODS ====================
    
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
    
    def find_node_by_value(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """Find node by value in the tree"""
        if not root:
            return None
        
        if root.val == val:
            return root
        
        left_result = self.find_node_by_value(root.left, val)
        if left_result:
            return left_result
        
        return self.find_node_by_value(root.right, val)
    
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
    print("=== Lowest Common Ancestor Demo ===\n")
    
    lca_solver = LowestCommonAncestor()
    
    # Build test tree
    arr = [3, 5, 1, 6, 2, 0, 8, None, None, 7, 4]
    root = lca_solver.build_tree_from_array(arr)
    
    print("Test Tree Structure:")
    lca_solver.print_tree_structure(root)
    print()
    
    # Find nodes for testing
    node_5 = lca_solver.find_node_by_value(root, 5)
    node_1 = lca_solver.find_node_by_value(root, 1)
    node_6 = lca_solver.find_node_by_value(root, 6)
    node_2 = lca_solver.find_node_by_value(root, 2)
    node_0 = lca_solver.find_node_by_value(root, 0)
    node_8 = lca_solver.find_node_by_value(root, 8)
    node_7 = lca_solver.find_node_by_value(root, 7)
    node_4 = lca_solver.find_node_by_value(root, 4)
    
    # Example 1: Basic LCA in binary tree
    print("1. Basic LCA in Binary Tree:")
    test_pairs = [
        (node_5, node_1, "5 and 1"),
        (node_5, node_4, "5 and 4"), 
        (node_6, node_2, "6 and 2"),
        (node_0, node_8, "0 and 8"),
        (node_7, node_4, "7 and 4")
    ]
    
    for p, q, description in test_pairs:
        if p and q:
            lca = lca_solver.lca_binary_tree_recursive(root, p, q)
            print(f"LCA of {description}: {lca.val if lca else None}")
    print()
    
    # Example 2: LCA using different methods
    print("2. LCA using Different Methods:")
    if node_5 and node_4:
        lca_recursive = lca_solver.lca_binary_tree_recursive(root, node_5, node_4)
        lca_paths = lca_solver.lca_using_paths(root, node_5, node_4)
        
        print(f"LCA of 5 and 4 (recursive): {lca_recursive.val if lca_recursive else None}")
        print(f"LCA of 5 and 4 (paths): {lca_paths.val if lca_paths else None}")
    print()
    
    # Example 3: Binary lifting LCA
    print("3. Binary Lifting LCA:")
    node_count = 11  # Approximate number of nodes
    lca_solver.build_binary_lifting_table(root, node_count)
    
    if node_5 and node_4:
        lca_binary_lifting = lca_solver.lca_binary_lifting(node_5, node_4)
        print(f"LCA of 5 and 4 (binary lifting): {lca_binary_lifting.val if lca_binary_lifting else None}")
    
    if node_6 and node_7:
        lca_binary_lifting = lca_solver.lca_binary_lifting(node_6, node_7)
        print(f"LCA of 6 and 7 (binary lifting): {lca_binary_lifting.val if lca_binary_lifting else None}")
    print()
    
    # Example 4: Euler tour + RMQ LCA
    print("4. Euler Tour + RMQ LCA:")
    lca_solver.build_euler_tour_rmq(root)
    
    if node_5 and node_4:
        lca_euler = lca_solver.lca_euler_rmq(node_5, node_4)
        print(f"LCA of 5 and 4 (Euler + RMQ): {lca_euler.val if lca_euler else None}")
    
    if node_6 and node_7:
        lca_euler = lca_solver.lca_euler_rmq(node_6, node_7)
        print(f"LCA of 6 and 7 (Euler + RMQ): {lca_euler.val if lca_euler else None}")
    print()
    
    # Example 5: Multiple queries
    print("5. Multiple LCA Queries:")
    queries = []
    if all([node_5, node_1, node_6, node_2, node_7, node_4]):
        queries = [
            (node_5, node_1),
            (node_6, node_2),
            (node_7, node_4)
        ]
    
    if queries:
        results_optimized = lca_solver.lca_multiple_queries_optimized(root, queries)
        print("Multiple query results:")
        for i, result in enumerate(results_optimized):
            print(f"  Query {i+1}: {result.val if result else None}")
    print()
    
    # Example 6: Special LCA problems
    print("6. Special LCA Problems:")
    
    # LCA of deepest leaves
    deepest_lca = lca_solver.lca_of_deepest_leaves(root)
    print(f"LCA of deepest leaves: {deepest_lca.val if deepest_lca else None}")
    
    # Distance between nodes
    if node_7 and node_4:
        distance = lca_solver.distance_between_nodes(root, node_7, node_4)
        print(f"Distance between 7 and 4: {distance}")
    
    # Kth ancestor
    if node_7:
        ancestor_1 = lca_solver.kth_ancestor(node_7, 1)  # Parent
        ancestor_2 = lca_solver.kth_ancestor(node_7, 2)  # Grandparent
        print(f"1st ancestor of 7: {ancestor_1.val if ancestor_1 else None}")
        print(f"2nd ancestor of 7: {ancestor_2.val if ancestor_2 else None}")
    print()
    
    # Example 7: BST LCA
    print("7. BST LCA (Building a BST):")
    # Build a simple BST for demonstration
    bst_arr = [6, 2, 8, 0, 4, 7, 9, None, None, 3, 5]
    bst_root = lca_solver.build_tree_from_array(bst_arr)
    
    print("BST Structure:")
    lca_solver.print_tree_structure(bst_root)
    
    bst_node_2 = lca_solver.find_node_by_value(bst_root, 2)
    bst_node_8 = lca_solver.find_node_by_value(bst_root, 8)
    bst_node_4 = lca_solver.find_node_by_value(bst_root, 4)
    bst_node_5 = lca_solver.find_node_by_value(bst_root, 5)
    
    if bst_node_2 and bst_node_8:
        bst_lca_recursive = lca_solver.lca_bst_recursive(bst_root, bst_node_2, bst_node_8)
        bst_lca_iterative = lca_solver.lca_bst_iterative(bst_root, bst_node_2, bst_node_8)
        print(f"BST LCA of 2 and 8 (recursive): {bst_lca_recursive.val if bst_lca_recursive else None}")
        print(f"BST LCA of 2 and 8 (iterative): {bst_lca_iterative.val if bst_lca_iterative else None}")
    
    if bst_node_4 and bst_node_5:
        bst_lca = lca_solver.lca_bst_recursive(bst_root, bst_node_4, bst_node_5)
        print(f"BST LCA of 4 and 5: {bst_lca.val if bst_lca else None}")
    
    print("\n=== Demo Complete ===") 