"""
Advanced Tree Algorithms - Sophisticated Tree Processing Techniques
This module implements advanced algorithms for complex tree operations and queries.
"""

from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, deque
import math

class TreeNode:
    """Binary tree node structure"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"

class BinaryLifting:
    """Binary Lifting for LCA and K-th Ancestor queries"""
    
    def __init__(self, adj_list: Dict[int, List[int]], root: int = 0):
        """
        Initialize binary lifting structure
        
        Time Complexity: O(n log n)
        Space Complexity: O(n log n)
        
        Args:
            adj_list: Adjacency list representation of tree
            root: Root node of tree
        """
        self.n = len(adj_list)
        self.root = root
        self.adj_list = adj_list
        
        # Calculate log(n) for binary lifting
        self.LOG = int(math.log2(self.n)) + 1
        
        # Initialize arrays
        self.parent = [[-1] * self.LOG for _ in range(self.n)]
        self.depth = [0] * self.n
        
        # Build the structure
        self._dfs(root, -1, 0)
        self._build_binary_lifting()
    
    def _dfs(self, node: int, par: int, d: int):
        """DFS to build parent and depth arrays"""
        self.parent[node][0] = par
        self.depth[node] = d
        
        for child in self.adj_list.get(node, []):
            if child != par:
                self._dfs(child, node, d + 1)
    
    def _build_binary_lifting(self):
        """Build binary lifting table"""
        for j in range(1, self.LOG):
            for i in range(self.n):
                if self.parent[i][j-1] != -1:
                    self.parent[i][j] = self.parent[self.parent[i][j-1]][j-1]
    
    def lca(self, u: int, v: int) -> int:
        """
        Find Lowest Common Ancestor of nodes u and v
        
        Time Complexity: O(log n)
        
        Args:
            u, v: Nodes to find LCA for
        
        Returns:
            int: LCA of u and v
        """
        # Make sure u is deeper than v
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        
        # Bring u to the same level as v
        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.parent[u][i]
        
        # If u and v are the same, return u
        if u == v:
            return u
        
        # Binary search for LCA
        for i in range(self.LOG - 1, -1, -1):
            if self.parent[u][i] != self.parent[v][i]:
                u = self.parent[u][i]
                v = self.parent[v][i]
        
        return self.parent[u][0]
    
    def kth_ancestor(self, node: int, k: int) -> int:
        """
        Find k-th ancestor of node
        
        Time Complexity: O(log n)
        
        Args:
            node: Starting node
            k: Number of steps up
        
        Returns:
            int: k-th ancestor or -1 if doesn't exist
        """
        if k > self.depth[node]:
            return -1
        
        for i in range(self.LOG):
            if (k >> i) & 1:
                node = self.parent[node][i]
                if node == -1:
                    return -1
        
        return node
    
    def distance(self, u: int, v: int) -> int:
        """
        Find distance between two nodes
        
        Args:
            u, v: Nodes to find distance between
        
        Returns:
            int: Distance between u and v
        """
        lca_node = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca_node]
    
    def is_ancestor(self, u: int, v: int) -> bool:
        """
        Check if u is ancestor of v
        
        Args:
            u, v: Nodes to check
        
        Returns:
            bool: True if u is ancestor of v
        """
        return self.lca(u, v) == u

class EulerTour:
    """Euler Tour Technique for subtree queries"""
    
    def __init__(self, adj_list: Dict[int, List[int]], root: int = 0):
        """
        Initialize Euler Tour
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            adj_list: Adjacency list representation
            root: Root node
        """
        self.n = len(adj_list)
        self.adj_list = adj_list
        self.root = root
        
        # Euler tour arrays
        self.tour = []
        self.first_occurrence = [-1] * self.n
        self.last_occurrence = [-1] * self.n
        
        # Subtree ranges
        self.subtree_start = [-1] * self.n
        self.subtree_end = [-1] * self.n
        
        # Build tour
        self._build_tour()
    
    def _build_tour(self):
        """Build Euler tour of the tree"""
        visited = set()
        
        def dfs(node: int, parent: int):
            visited.add(node)
            
            # First occurrence
            self.first_occurrence[node] = len(self.tour)
            self.subtree_start[node] = len(self.tour)
            self.tour.append(node)
            
            for child in self.adj_list.get(node, []):
                if child not in visited:
                    dfs(child, node)
                    self.tour.append(node)  # Return to parent
            
            # Last occurrence
            self.last_occurrence[node] = len(self.tour) - 1
            self.subtree_end[node] = len(self.tour) - 1
        
        dfs(self.root, -1)
    
    def get_subtree_range(self, node: int) -> Tuple[int, int]:
        """
        Get range in tour representing subtree of node
        
        Args:
            node: Node to get subtree range for
        
        Returns:
            Tuple of (start_index, end_index)
        """
        return self.subtree_start[node], self.subtree_end[node]
    
    def is_ancestor(self, u: int, v: int) -> bool:
        """
        Check if u is ancestor of v using Euler tour
        
        Args:
            u, v: Nodes to check
        
        Returns:
            bool: True if u is ancestor of v
        """
        u_start, u_end = self.get_subtree_range(u)
        v_start, v_end = self.get_subtree_range(v)
        
        return u_start <= v_start and v_end <= u_end

class EulerTourWithUpdates:
    """Euler Tour with support for subtree updates and queries"""
    
    def __init__(self, adj_list: Dict[int, List[int]], node_values: List[int], root: int = 0):
        """
        Initialize Euler Tour with values
        
        Args:
            adj_list: Adjacency list
            node_values: Initial values of nodes
            root: Root node
        """
        self.euler_tour = EulerTour(adj_list, root)
        self.values = node_values[:]
        
        # Segment tree for range updates and queries
        tour_size = len(self.euler_tour.tour)
        self.seg_tree = [0] * (4 * tour_size)
        self.lazy = [0] * (4 * tour_size)
    
    def update_subtree(self, node: int, delta: int):
        """
        Add delta to all nodes in subtree
        
        Time Complexity: O(log n)
        
        Args:
            node: Root of subtree
            delta: Value to add
        """
        start, end = self.euler_tour.get_subtree_range(node)
        self._update_range(1, 0, len(self.euler_tour.tour) - 1, start, end, delta)
    
    def query_subtree_sum(self, node: int) -> int:
        """
        Get sum of all nodes in subtree
        
        Time Complexity: O(log n)
        
        Args:
            node: Root of subtree
        
        Returns:
            int: Sum of subtree
        """
        start, end = self.euler_tour.get_subtree_range(node)
        return self._query_range(1, 0, len(self.euler_tour.tour) - 1, start, end)
    
    def _push(self, node: int, start: int, end: int):
        """Push lazy propagation"""
        if self.lazy[node] != 0:
            self.seg_tree[node] += self.lazy[node] * (end - start + 1)
            
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            
            self.lazy[node] = 0
    
    def _update_range(self, node: int, start: int, end: int, l: int, r: int, delta: int):
        """Range update with lazy propagation"""
        self._push(node, start, end)
        
        if start > r or end < l:
            return
        
        if start >= l and end <= r:
            self.lazy[node] += delta
            self._push(node, start, end)
            return
        
        mid = (start + end) // 2
        self._update_range(2 * node, start, mid, l, r, delta)
        self._update_range(2 * node + 1, mid + 1, end, l, r, delta)
        
        self._push(2 * node, start, mid)
        self._push(2 * node + 1, mid + 1, end)
        
        self.seg_tree[node] = self.seg_tree[2 * node] + self.seg_tree[2 * node + 1]
    
    def _query_range(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Range query with lazy propagation"""
        if start > r or end < l:
            return 0
        
        self._push(node, start, end)
        
        if start >= l and end <= r:
            return self.seg_tree[node]
        
        mid = (start + end) // 2
        left_sum = self._query_range(2 * node, start, mid, l, r)
        right_sum = self._query_range(2 * node + 1, mid + 1, end, l, r)
        
        return left_sum + right_sum

class CentroidDecomposition:
    """Centroid Decomposition for tree path queries"""
    
    def __init__(self, adj_list: Dict[int, List[int]]):
        """
        Initialize centroid decomposition
        
        Time Complexity: O(n log n)
        Space Complexity: O(n log n)
        
        Args:
            adj_list: Adjacency list representation
        """
        self.adj_list = adj_list
        self.n = len(adj_list)
        
        # Centroid tree structure
        self.centroid_parent = [-1] * self.n
        self.centroid_children = [[] for _ in range(self.n)]
        
        # Tree properties
        self.removed = [False] * self.n
        self.subtree_size = [0] * self.n
        
        # Build centroid decomposition
        self.centroid_root = self._decompose(0)
    
    def _get_subtree_size(self, node: int, parent: int) -> int:
        """Calculate subtree sizes"""
        self.subtree_size[node] = 1
        
        for child in self.adj_list.get(node, []):
            if child != parent and not self.removed[child]:
                self.subtree_size[node] += self._get_subtree_size(child, node)
        
        return self.subtree_size[node]
    
    def _find_centroid(self, node: int, parent: int, tree_size: int) -> int:
        """Find centroid of current tree"""
        for child in self.adj_list.get(node, []):
            if (child != parent and not self.removed[child] and 
                self.subtree_size[child] > tree_size // 2):
                return self._find_centroid(child, node, tree_size)
        
        return node
    
    def _decompose(self, node: int) -> int:
        """Recursively decompose tree"""
        tree_size = self._get_subtree_size(node, -1)
        centroid = self._find_centroid(node, -1, tree_size)
        
        self.removed[centroid] = True
        
        for child in self.adj_list.get(centroid, []):
            if not self.removed[child]:
                child_centroid = self._decompose(child)
                self.centroid_parent[child_centroid] = centroid
                self.centroid_children[centroid].append(child_centroid)
        
        return centroid
    
    def query_path_to_centroid(self, node: int, centroid: int) -> List[int]:
        """
        Get path from node to centroid in original tree
        
        Args:
            node: Starting node
            centroid: Target centroid
        
        Returns:
            List of nodes in path
        """
        path = []
        visited = set()
        
        def dfs(current: int, target: int, parent: int) -> bool:
            if current in visited:
                return False
            
            visited.add(current)
            path.append(current)
            
            if current == target:
                return True
            
            for neighbor in self.adj_list.get(current, []):
                if neighbor != parent and not self.removed[neighbor]:
                    if dfs(neighbor, target, current):
                        return True
            
            path.pop()
            return False
        
        # Temporarily unrestrict removed nodes for path finding
        original_removed = self.removed[:]
        self.removed = [False] * self.n
        
        dfs(node, centroid, -1)
        
        # Restore removed status
        self.removed = original_removed
        
        return path
    
    def distance_query(self, u: int, v: int) -> int:
        """
        Query distance between two nodes using centroid decomposition
        
        Time Complexity: O(log n) per query
        
        Args:
            u, v: Nodes to find distance between
        
        Returns:
            int: Distance between u and v
        """
        # Find LCA in centroid tree
        ancestors_u = []
        current = u
        while current != -1:
            ancestors_u.append(current)
            current = self.centroid_parent[current]
        
        ancestors_v = []
        current = v
        while current != -1:
            ancestors_v.append(current)
            current = self.centroid_parent[current]
        
        # Find common ancestor
        set_u = set(ancestors_u)
        lca_centroid = None
        
        for ancestor in ancestors_v:
            if ancestor in set_u:
                lca_centroid = ancestor
                break
        
        if lca_centroid is None:
            return -1
        
        # Calculate distance through LCA centroid
        path_u = self.query_path_to_centroid(u, lca_centroid)
        path_v = self.query_path_to_centroid(v, lca_centroid)
        
        return len(path_u) + len(path_v) - 2  # Subtract 2 because LCA is counted twice

class HeavyLightDecomposition:
    """Heavy-Light Decomposition for path queries"""
    
    def __init__(self, adj_list: Dict[int, List[int]], node_values: List[int], root: int = 0):
        """
        Initialize Heavy-Light Decomposition
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Args:
            adj_list: Adjacency list representation
            node_values: Values of nodes
            root: Root of tree
        """
        self.adj_list = adj_list
        self.n = len(adj_list)
        self.root = root
        self.values = node_values
        
        # HLD arrays
        self.parent = [-1] * self.n
        self.depth = [0] * self.n
        self.subtree_size = [0] * self.n
        self.heavy_child = [-1] * self.n
        
        # Chain information
        self.chain_head = [-1] * self.n
        self.chain_pos = [-1] * self.n
        self.chains = []
        
        # Build HLD
        self._dfs1(root, -1, 0)
        self._dfs2(root, root)
        
        # Segment tree for each chain
        self.seg_trees = []
        for chain in self.chains:
            chain_values = [self.values[node] for node in chain]
            self.seg_trees.append(SegmentTreeForHLD(chain_values))
    
    def _dfs1(self, node: int, par: int, d: int):
        """First DFS to calculate subtree sizes and find heavy children"""
        self.parent[node] = par
        self.depth[node] = d
        self.subtree_size[node] = 1
        
        max_child_size = 0
        
        for child in self.adj_list.get(node, []):
            if child != par:
                self._dfs1(child, node, d + 1)
                self.subtree_size[node] += self.subtree_size[child]
                
                if self.subtree_size[child] > max_child_size:
                    max_child_size = self.subtree_size[child]
                    self.heavy_child[node] = child
    
    def _dfs2(self, node: int, head: int):
        """Second DFS to build chains"""
        self.chain_head[node] = head
        
        # If this is the start of a new chain
        if head == node:
            self.chains.append([])
        
        # Add node to current chain
        chain_id = len(self.chains) - 1
        self.chain_pos[node] = len(self.chains[chain_id])
        self.chains[chain_id].append(node)
        
        # Continue heavy edge first
        if self.heavy_child[node] != -1:
            self._dfs2(self.heavy_child[node], head)
        
        # Process light children (start new chains)
        for child in self.adj_list.get(node, []):
            if child != self.parent[node] and child != self.heavy_child[node]:
                self._dfs2(child, child)
    
    def _get_chain_id(self, node: int) -> int:
        """Get chain ID for a node"""
        head = self.chain_head[node]
        for i, chain in enumerate(self.chains):
            if chain and chain[0] == head:
                return i
        return -1
    
    def query_path(self, u: int, v: int) -> int:
        """
        Query sum on path from u to v
        
        Time Complexity: O(log^2 n)
        
        Args:
            u, v: Endpoints of path
        
        Returns:
            int: Sum of values on path
        """
        total_sum = 0
        
        while self.chain_head[u] != self.chain_head[v]:
            # Make sure u is deeper
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            
            # Query from u to head of its chain
            chain_id = self._get_chain_id(u)
            head_pos = self.chain_pos[self.chain_head[u]]
            u_pos = self.chain_pos[u]
            
            total_sum += self.seg_trees[chain_id].query(head_pos, u_pos)
            
            # Move to parent of chain head
            u = self.parent[self.chain_head[u]]
        
        # Now u and v are in the same chain
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        
        chain_id = self._get_chain_id(u)
        u_pos = self.chain_pos[u]
        v_pos = self.chain_pos[v]
        
        total_sum += self.seg_trees[chain_id].query(u_pos, v_pos)
        
        return total_sum
    
    def update_node(self, node: int, new_value: int):
        """
        Update value of a single node
        
        Time Complexity: O(log n)
        
        Args:
            node: Node to update
            new_value: New value
        """
        chain_id = self._get_chain_id(node)
        pos = self.chain_pos[node]
        
        self.seg_trees[chain_id].update(pos, new_value)
        self.values[node] = new_value

class SegmentTreeForHLD:
    """Simple segment tree for HLD"""
    
    def __init__(self, arr: List[int]):
        """Initialize segment tree"""
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.arr = arr[:]
        self._build(1, 0, self.n - 1)
    
    def _build(self, node: int, start: int, end: int):
        """Build segment tree"""
        if start == end:
            self.tree[node] = self.arr[start]
        else:
            mid = (start + end) // 2
            self._build(2 * node, start, mid)
            self._build(2 * node + 1, mid + 1, end)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def update(self, pos: int, new_val: int):
        """Update single position"""
        self._update(1, 0, self.n - 1, pos, new_val)
    
    def _update(self, node: int, start: int, end: int, pos: int, new_val: int):
        """Recursive update"""
        if start == end:
            self.tree[node] = new_val
            self.arr[pos] = new_val
        else:
            mid = (start + end) // 2
            if pos <= mid:
                self._update(2 * node, start, mid, pos, new_val)
            else:
                self._update(2 * node + 1, mid + 1, end, pos, new_val)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def query(self, left: int, right: int) -> int:
        """Query range sum"""
        return self._query(1, 0, self.n - 1, left, right)
    
    def _query(self, node: int, start: int, end: int, left: int, right: int) -> int:
        """Recursive range query"""
        if start > right or end < left:
            return 0
        
        if start >= left and end <= right:
            return self.tree[node]
        
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))

class TreeFlattening:
    """Flatten tree for efficient range queries"""
    
    def __init__(self, adj_list: Dict[int, List[int]], node_values: List[int], root: int = 0):
        """
        Initialize tree flattening
        
        Args:
            adj_list: Adjacency list
            node_values: Values of nodes
            root: Root node
        """
        self.adj_list = adj_list
        self.n = len(adj_list)
        self.values = node_values
        self.root = root
        
        # Flattening arrays
        self.flat_array = []
        self.node_to_index = [-1] * self.n
        self.subtree_start = [-1] * self.n
        self.subtree_end = [-1] * self.n
        
        # Build flattened representation
        self._flatten()
        
        # Segment tree on flattened array
        self.seg_tree = SegmentTreeForHLD(self.flat_array)
    
    def _flatten(self):
        """Flatten tree using DFS"""
        visited = set()
        
        def dfs(node: int, parent: int):
            visited.add(node)
            
            # Record start of subtree
            self.subtree_start[node] = len(self.flat_array)
            self.node_to_index[node] = len(self.flat_array)
            self.flat_array.append(self.values[node])
            
            # Process children
            for child in self.adj_list.get(node, []):
                if child not in visited:
                    dfs(child, node)
            
            # Record end of subtree
            self.subtree_end[node] = len(self.flat_array) - 1
        
        dfs(self.root, -1)
    
    def query_subtree(self, node: int) -> int:
        """
        Query sum of subtree rooted at node
        
        Time Complexity: O(log n)
        
        Args:
            node: Root of subtree
        
        Returns:
            int: Sum of subtree
        """
        start = self.subtree_start[node]
        end = self.subtree_end[node]
        return self.seg_tree.query(start, end)
    
    def update_node(self, node: int, new_value: int):
        """
        Update value of node
        
        Time Complexity: O(log n)
        
        Args:
            node: Node to update
            new_value: New value
        """
        index = self.node_to_index[node]
        self.seg_tree.update(index, new_value)
        self.values[node] = new_value
    
    def update_subtree(self, node: int, delta: int):
        """
        Add delta to all nodes in subtree
        Note: This requires a segment tree with range updates
        
        Args:
            node: Root of subtree
            delta: Value to add
        """
        # This would require lazy propagation segment tree
        # For now, update each node individually (less efficient)
        start = self.subtree_start[node]
        end = self.subtree_end[node]
        
        for i in range(start, end + 1):
            current_value = self.flat_array[i]
            self.seg_tree.update(i, current_value + delta)

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Advanced Tree Algorithms Demo ===\n")
    
    # Create a sample tree
    # Tree structure:
    #       0
    #      / \
    #     1   2
    #    /|   |\
    #   3 4   5 6
    #  /      |
    # 7       8
    
    adj_list = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5, 6],
        3: [1, 7],
        4: [1],
        5: [2, 8],
        6: [2],
        7: [3],
        8: [5]
    }
    
    node_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # Example 1: Binary Lifting
    print("1. Binary Lifting (LCA and K-th Ancestor):")
    
    binary_lifting = BinaryLifting(adj_list, root=0)
    
    # Test LCA queries
    test_pairs = [(7, 8), (3, 4), (5, 6), (7, 6)]
    for u, v in test_pairs:
        lca = binary_lifting.lca(u, v)
        distance = binary_lifting.distance(u, v)
        print(f"LCA({u}, {v}) = {lca}, Distance = {distance}")
    
    # Test k-th ancestor queries
    test_ancestors = [(7, 1), (7, 2), (8, 3), (4, 1)]
    for node, k in test_ancestors:
        ancestor = binary_lifting.kth_ancestor(node, k)
        print(f"{k}-th ancestor of {node} = {ancestor}")
    
    # Test ancestor relationship
    ancestor_tests = [(0, 7), (1, 3), (2, 8), (3, 8)]
    for u, v in ancestor_tests:
        is_anc = binary_lifting.is_ancestor(u, v)
        print(f"Is {u} ancestor of {v}? {is_anc}")
    print()
    
    # Example 2: Euler Tour Technique
    print("2. Euler Tour Technique:")
    
    euler_tour = EulerTour(adj_list, root=0)
    
    print(f"Euler tour: {euler_tour.tour}")
    
    # Test subtree ranges
    for node in range(len(adj_list)):
        start, end = euler_tour.get_subtree_range(node)
        print(f"Node {node} subtree range: [{start}, {end}]")
    
    # Test ancestor queries using Euler tour
    print("Ancestor tests using Euler tour:")
    for u, v in ancestor_tests:
        is_anc = euler_tour.is_ancestor(u, v)
        print(f"Is {u} ancestor of {v}? {is_anc}")
    print()
    
    # Example 3: Euler Tour with Updates
    print("3. Euler Tour with Subtree Updates:")
    
    euler_updates = EulerTourWithUpdates(adj_list, node_values, root=0)
    
    # Test subtree queries
    for node in [0, 1, 2, 3]:
        subtree_sum = euler_updates.query_subtree_sum(node)
        print(f"Subtree sum of node {node}: {subtree_sum}")
    
    # Update subtree and query again
    print("After adding 100 to subtree of node 1:")
    euler_updates.update_subtree(1, 100)
    
    for node in [0, 1, 2, 3]:
        subtree_sum = euler_updates.query_subtree_sum(node)
        print(f"Subtree sum of node {node}: {subtree_sum}")
    print()
    
    # Example 4: Centroid Decomposition
    print("4. Centroid Decomposition:")
    
    centroid_decomp = CentroidDecomposition(adj_list)
    
    print(f"Centroid root: {centroid_decomp.centroid_root}")
    print("Centroid tree structure:")
    for i in range(len(adj_list)):
        parent = centroid_decomp.centroid_parent[i]
        children = centroid_decomp.centroid_children[i]
        print(f"  Node {i}: parent={parent}, children={children}")
    
    # Test distance queries
    print("Distance queries using centroid decomposition:")
    for u, v in test_pairs:
        distance = centroid_decomp.distance_query(u, v)
        print(f"Distance({u}, {v}) = {distance}")
    print()
    
    # Example 5: Heavy-Light Decomposition
    print("5. Heavy-Light Decomposition:")
    
    hld = HeavyLightDecomposition(adj_list, node_values, root=0)
    
    print("Heavy-Light chains:")
    for i, chain in enumerate(hld.chains):
        print(f"  Chain {i}: {chain}")
    
    # Test path queries
    print("Path sum queries:")
    for u, v in test_pairs:
        path_sum = hld.query_path(u, v)
        print(f"Path sum({u}, {v}) = {path_sum}")
    
    # Test node updates
    print("After updating node 3 to value 1000:")
    hld.update_node(3, 1000)
    
    for u, v in [(7, 8), (3, 4)]:
        path_sum = hld.query_path(u, v)
        print(f"Path sum({u}, {v}) = {path_sum}")
    print()
    
    # Example 6: Tree Flattening
    print("6. Tree Flattening for Range Queries:")
    
    tree_flatten = TreeFlattening(adj_list, node_values, root=0)
    
    print(f"Flattened array: {tree_flatten.flat_array}")
    print("Node to index mapping:")
    for node in range(len(adj_list)):
        index = tree_flatten.node_to_index[node]
        start = tree_flatten.subtree_start[node]
        end = tree_flatten.subtree_end[node]
        print(f"  Node {node}: index={index}, subtree=[{start}, {end}]")
    
    # Test subtree queries
    print("Subtree sum queries:")
    for node in [0, 1, 2, 3, 5]:
        subtree_sum = tree_flatten.query_subtree(node)
        print(f"Subtree sum of node {node}: {subtree_sum}")
    
    # Test node updates
    print("After updating node 5 to value 600:")
    tree_flatten.update_node(5, 600)
    
    for node in [0, 2, 5]:
        subtree_sum = tree_flatten.query_subtree(node)
        print(f"Subtree sum of node {node}: {subtree_sum}")
    print()
    
    # Example 7: Performance Comparison
    print("7. Performance Analysis:")
    
    # Create larger tree for performance testing
    large_adj_list = {}
    large_values = []
    
    # Create a balanced binary tree
    n = 127  # 2^7 - 1 nodes
    for i in range(n):
        large_adj_list[i] = []
        large_values.append(i + 1)
        
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        if left_child < n:
            large_adj_list[i].append(left_child)
            large_adj_list[left_child] = large_adj_list.get(left_child, []) + [i]
        
        if right_child < n:
            large_adj_list[i].append(right_child)
            large_adj_list[right_child] = large_adj_list.get(right_child, []) + [i]
    
    print(f"Testing with tree of {n} nodes")
    
    # Test different algorithms
    large_binary_lifting = BinaryLifting(large_adj_list, root=0)
    large_hld = HeavyLightDecomposition(large_adj_list, large_values, root=0)
    large_flatten = TreeFlattening(large_adj_list, large_values, root=0)
    
    # Sample queries
    test_nodes = [50, 75, 100, 120]
    
    print("Sample LCA queries:")
    for i in range(len(test_nodes) - 1):
        u, v = test_nodes[i], test_nodes[i + 1]
        lca = large_binary_lifting.lca(u, v)
        print(f"  LCA({u}, {v}) = {lca}")
    
    print("Sample path sum queries (HLD):")
    for i in range(len(test_nodes) - 1):
        u, v = test_nodes[i], test_nodes[i + 1]
        path_sum = large_hld.query_path(u, v)
        print(f"  Path sum({u}, {v}) = {path_sum}")
    
    print("Sample subtree sum queries (Flattening):")
    for node in test_nodes[:3]:
        subtree_sum = large_flatten.query_subtree(node)
        print(f"  Subtree sum({node}) = {subtree_sum}")
    
    print("\n=== Demo Complete ===") 