"""
Advanced Tree Techniques and Research Topics
===========================================

Topics: Persistent trees, lazy propagation, tree isomorphism, advanced algorithms
Companies: Research labs, advanced tech companies, competitive programming
Difficulty: Expert to Research level
Time Complexity: Varies (O(log n) to O(n²) depending on technique)
Space Complexity: O(n) to O(n log n) for persistent structures
"""

from typing import List, Optional, Dict, Any, Tuple, Union, Set
from collections import defaultdict, deque
import copy
import hashlib
import time

class AdvancedTreeTechniques:
    
    def __init__(self):
        """Initialize with advanced technique tracking"""
        self.techniques_implemented = 0
        self.research_topics_covered = 0
    
    # ==========================================
    # 1. PERSISTENT TREE STRUCTURES
    # ==========================================
    
    def explain_persistent_trees(self) -> None:
        """
        Explain persistent (functional) tree data structures
        """
        print("=== PERSISTENT TREE STRUCTURES ===")
        print("Immutable trees with efficient versioning")
        print()
        print("PERSISTENCE CONCEPTS:")
        print("• Immutable data structures: Never modify existing nodes")
        print("• Structural sharing: Share unchanged parts between versions")
        print("• Path copying: Copy only the path to modification")
        print("• Version control: Maintain multiple versions efficiently")
        print()
        print("TYPES OF PERSISTENCE:")
        print("• Partially persistent: Access any version, modify only latest")
        print("• Fully persistent: Access and modify any version")
        print("• Confluently persistent: Merge different versions")
        print("• Functional persistence: Immutable by design")
        print()
        print("ADVANTAGES:")
        print("• Thread safety: Natural concurrency support")
        print("• Undo/redo operations: Version history maintenance")
        print("• Debugging: State history for analysis")
        print("• Functional programming: Natural fit for FP paradigms")
        print()
        print("IMPLEMENTATION TECHNIQUES:")
        print("• Copy-on-write: Share until modification needed")
        print("• Fat node method: Store all versions in nodes")
        print("• Node splitting: Split nodes when capacity exceeded")
        print("• Lazy evaluation: Defer computations until needed")
        print()
        print("APPLICATIONS:")
        print("• Version control systems (Git internals)")
        print("• Functional programming languages")
        print("• Database snapshots and time-travel queries")
        print("• Undo mechanisms in editors and applications")
    
    def demonstrate_persistent_trees(self) -> None:
        """
        Demonstrate persistent tree implementations
        """
        print("=== PERSISTENT TREE DEMONSTRATIONS ===")
        
        print("1. PERSISTENT BINARY SEARCH TREE")
        print("   Immutable BST with structural sharing")
        
        # Create initial tree
        tree_v1 = PersistentBST()
        values = [50, 30, 70, 20, 40, 60, 80]
        
        for val in values:
            tree_v1 = tree_v1.insert(val)
        
        print(f"   Version 1 - Inserted: {values}")
        print(f"   Tree size: {tree_v1.size()}")
        
        # Create new version by inserting
        tree_v2 = tree_v1.insert(25)
        tree_v2 = tree_v2.insert(75)
        
        print(f"   Version 2 - Added 25 and 75")
        print(f"   V1 size: {tree_v1.size()}, V2 size: {tree_v2.size()}")
        
        # Create new version by deleting
        tree_v3 = tree_v2.delete(30)
        
        print(f"   Version 3 - Deleted 30")
        print(f"   V2 size: {tree_v2.size()}, V3 size: {tree_v3.size()}")
        
        # Verify all versions are independent
        print("   Verification - searching for 30:")
        print(f"     V1 contains 30: {tree_v1.contains(30)}")
        print(f"     V2 contains 30: {tree_v2.contains(30)}")
        print(f"     V3 contains 30: {tree_v3.contains(30)}")
        
        # Show memory sharing
        shared_nodes = tree_v1.count_shared_nodes(tree_v2)
        print(f"   Shared nodes between V1 and V2: {shared_nodes}")
        print()
        
        print("2. PERSISTENT ARRAY (TRIE-BASED)")
        print("   Immutable array using trie structure")
        
        # Create persistent array
        arr_v1 = PersistentArray()
        
        # Add elements
        for i in range(8):
            arr_v1 = arr_v1.set(i, i * 10)
        
        print(f"   Version 1: {[arr_v1.get(i) for i in range(8)]}")
        
        # Create new version with modifications
        arr_v2 = arr_v1.set(3, 999).set(5, 888)
        
        print(f"   Version 2: {[arr_v2.get(i) for i in range(8)]}")
        print(f"   Original : {[arr_v1.get(i) for i in range(8)]}")
        
        # Show structural sharing efficiency
        nodes_v1 = arr_v1.count_nodes()
        nodes_v2 = arr_v2.count_nodes()
        print(f"   Nodes in V1: {nodes_v1}, Nodes in V2: {nodes_v2}")
        print(f"   Sharing efficiency: {((nodes_v1 + nodes_v2 - arr_v2.count_unique_nodes()) / (nodes_v1 + nodes_v2)) * 100:.1f}%")


class PersistentBSTNode:
    """Node in persistent BST"""
    
    def __init__(self, value: int, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        self.node_id = id(self)  # For tracking sharing
    
    def __str__(self):
        return f"Node({self.value})"


class PersistentBST:
    """
    Persistent Binary Search Tree with structural sharing
    
    Immutable BST where modifications create new versions
    """
    
    def __init__(self, root: Optional[PersistentBSTNode] = None):
        self.root = root
        self._size = 0 if root is None else self._calculate_size(root)
    
    def insert(self, value: int) -> 'PersistentBST':
        """Insert value, returning new tree version"""
        new_root = self._insert_recursive(self.root, value)
        return PersistentBST(new_root)
    
    def _insert_recursive(self, node: Optional[PersistentBSTNode], value: int) -> PersistentBSTNode:
        """Recursive insertion with path copying"""
        if not node:
            return PersistentBSTNode(value)
        
        if value < node.value:
            # Create new node with new left child, sharing right subtree
            new_left = self._insert_recursive(node.left, value)
            return PersistentBSTNode(node.value, new_left, node.right)
        elif value > node.value:
            # Create new node with new right child, sharing left subtree
            new_right = self._insert_recursive(node.right, value)
            return PersistentBSTNode(node.value, node.left, new_right)
        else:
            # Value already exists, return existing node
            return node
    
    def delete(self, value: int) -> 'PersistentBST':
        """Delete value, returning new tree version"""
        new_root = self._delete_recursive(self.root, value)
        return PersistentBST(new_root)
    
    def _delete_recursive(self, node: Optional[PersistentBSTNode], value: int) -> Optional[PersistentBSTNode]:
        """Recursive deletion with path copying"""
        if not node:
            return None
        
        if value < node.value:
            new_left = self._delete_recursive(node.left, value)
            return PersistentBSTNode(node.value, new_left, node.right)
        elif value > node.value:
            new_right = self._delete_recursive(node.right, value)
            return PersistentBSTNode(node.value, node.left, new_right)
        else:
            # Node to delete found
            if not node.left:
                return node.right
            elif not node.right:
                return node.left
            else:
                # Node with two children - find inorder successor
                successor = self._find_min(node.right)
                new_right = self._delete_recursive(node.right, successor.value)
                return PersistentBSTNode(successor.value, node.left, new_right)
    
    def _find_min(self, node: PersistentBSTNode) -> PersistentBSTNode:
        """Find minimum value node"""
        while node.left:
            node = node.left
        return node
    
    def contains(self, value: int) -> bool:
        """Check if value exists in tree"""
        return self._contains_recursive(self.root, value)
    
    def _contains_recursive(self, node: Optional[PersistentBSTNode], value: int) -> bool:
        """Recursive search"""
        if not node:
            return False
        
        if value == node.value:
            return True
        elif value < node.value:
            return self._contains_recursive(node.left, value)
        else:
            return self._contains_recursive(node.right, value)
    
    def size(self) -> int:
        """Get tree size"""
        return self._calculate_size(self.root) if self.root else 0
    
    def _calculate_size(self, node: Optional[PersistentBSTNode]) -> int:
        """Calculate tree size"""
        if not node:
            return 0
        return 1 + self._calculate_size(node.left) + self._calculate_size(node.right)
    
    def count_shared_nodes(self, other: 'PersistentBST') -> int:
        """Count nodes shared between two tree versions"""
        self_nodes = set()
        other_nodes = set()
        
        self._collect_node_ids(self.root, self_nodes)
        self._collect_node_ids(other.root, other_nodes)
        
        return len(self_nodes.intersection(other_nodes))
    
    def _collect_node_ids(self, node: Optional[PersistentBSTNode], node_set: Set[int]) -> None:
        """Collect node IDs for sharing analysis"""
        if node:
            node_set.add(node.node_id)
            self._collect_node_ids(node.left, node_set)
            self._collect_node_ids(node.right, node_set)


class PersistentArrayNode:
    """Node in persistent array trie"""
    
    def __init__(self, children=None, value=None):
        self.children = children or {}  # Dict[int, PersistentArrayNode]
        self.value = value
        self.node_id = id(self)


class PersistentArray:
    """
    Persistent array using trie structure
    
    Efficient random access with structural sharing
    """
    
    def __init__(self, root: Optional[PersistentArrayNode] = None, branching_factor: int = 4):
        self.root = root or PersistentArrayNode()
        self.branching_factor = branching_factor
        self.bits_per_level = branching_factor.bit_length() - 1
    
    def get(self, index: int) -> Optional[Any]:
        """Get value at index"""
        node = self.root
        
        # Navigate down the trie
        for level in range(self._get_depth(index), -1, -1):
            child_index = (index >> (level * self.bits_per_level)) & ((1 << self.bits_per_level) - 1)
            
            if child_index not in node.children:
                return None
            
            node = node.children[child_index]
        
        return node.value
    
    def set(self, index: int, value: Any) -> 'PersistentArray':
        """Set value at index, returning new array version"""
        new_root = self._set_recursive(self.root, index, value, self._get_depth(index))
        return PersistentArray(new_root, self.branching_factor)
    
    def _set_recursive(self, node: PersistentArrayNode, index: int, value: Any, level: int) -> PersistentArrayNode:
        """Recursive set with path copying"""
        if level < 0:
            # Leaf level - create new node with value
            return PersistentArrayNode(value=value)
        
        child_index = (index >> (level * self.bits_per_level)) & ((1 << self.bits_per_level) - 1)
        
        # Create new node copying existing children
        new_children = node.children.copy()
        
        # Get or create child node
        child = node.children.get(child_index, PersistentArrayNode())
        new_child = self._set_recursive(child, index, value, level - 1)
        new_children[child_index] = new_child
        
        return PersistentArrayNode(new_children, node.value)
    
    def _get_depth(self, index: int) -> int:
        """Calculate depth needed for index"""
        if index == 0:
            return 0
        
        depth = 0
        while (1 << ((depth + 1) * self.bits_per_level)) <= index:
            depth += 1
        
        return depth
    
    def count_nodes(self) -> int:
        """Count total nodes in array"""
        return self._count_nodes_recursive(self.root)
    
    def _count_nodes_recursive(self, node: PersistentArrayNode) -> int:
        """Recursive node counting"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes_recursive(child)
        return count
    
    def count_unique_nodes(self) -> int:
        """Count unique nodes (for sharing analysis)"""
        seen = set()
        self._collect_unique_nodes(self.root, seen)
        return len(seen)
    
    def _collect_unique_nodes(self, node: PersistentArrayNode, seen: Set[int]) -> None:
        """Collect unique node IDs"""
        if node.node_id not in seen:
            seen.add(node.node_id)
            for child in node.children.values():
                self._collect_unique_nodes(child, seen)


# ==========================================
# 2. LAZY PROPAGATION IN TREES
# ==========================================

class LazyPropagationTrees:
    """
    Lazy propagation techniques for efficient range updates
    """
    
    def explain_lazy_propagation(self) -> None:
        """Explain lazy propagation concepts"""
        print("=== LAZY PROPAGATION IN TREES ===")
        print("Efficient range updates using deferred computation")
        print()
        print("LAZY PROPAGATION CONCEPT:")
        print("• Defer updates until absolutely necessary")
        print("• Mark nodes with pending updates")
        print("• Propagate updates only when accessing children")
        print("• Batch multiple updates for efficiency")
        print()
        print("APPLICATIONS:")
        print("• Segment trees with range updates")
        print("• Binary indexed trees with lazy updates")
        print("• Tree-based range query structures")
        print("• Dynamic programming on trees")
        print()
        print("UPDATE PATTERNS:")
        print("• Range addition: Add value to all elements in range")
        print("• Range assignment: Set all elements in range to value")
        print("• Range multiplication: Multiply all elements by factor")
        print("• Complex updates: Composition of multiple operations")
        print()
        print("IMPLEMENTATION STRATEGIES:")
        print("• Lazy flags: Boolean indicators for pending updates")
        print("• Update values: Store pending update operations")
        print("• Push operations: Propagate updates to children")
        print("• Combine operations: Merge multiple pending updates")
    
    def demonstrate_lazy_propagation(self) -> None:
        """Demonstrate lazy propagation implementations"""
        print("=== LAZY PROPAGATION DEMONSTRATIONS ===")
        
        print("1. LAZY SEGMENT TREE")
        print("   Range updates with O(log n) complexity")
        
        # Create array and segment tree
        arr = [1, 3, 5, 7, 9, 11, 13, 15]
        lazy_tree = LazySegmentTree(arr)
        
        print(f"   Initial array: {arr}")
        print(f"   Sum of range [2, 5]: {lazy_tree.range_query(2, 5)}")
        
        # Range update
        lazy_tree.range_update(1, 4, 10)  # Add 10 to elements 1-4
        print(f"   After adding 10 to range [1, 4]:")
        print(f"   Sum of range [2, 5]: {lazy_tree.range_query(2, 5)}")
        print(f"   Sum of range [0, 7]: {lazy_tree.range_query(0, 7)}")
        
        # Another range update
        lazy_tree.range_update(3, 6, 5)   # Add 5 to elements 3-6
        print(f"   After adding 5 to range [3, 6]:")
        print(f"   Sum of range [2, 5]: {lazy_tree.range_query(2, 5)}")
        
        # Show lazy propagation efficiency
        print(f"   Operations performed: {lazy_tree.operations_count}")
        print(f"   Nodes touched: {lazy_tree.nodes_touched}")
        print()
        
        print("2. LAZY PROPAGATION IN TREE DP")
        print("   Dynamic programming with lazy updates")
        
        # Create tree for DP
        tree_dp = TreeDPWithLazy(8)  # 8 nodes
        
        # Add edges to form tree
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (5, 7)]
        for u, v in edges:
            tree_dp.add_edge(u, v)
        
        # Initialize values
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        for i, val in enumerate(values):
            tree_dp.set_value(i, val)
        
        print(f"   Tree with {tree_dp.n} nodes")
        print(f"   Initial values: {values}")
        
        # Subtree update
        tree_dp.subtree_update(0, 10)  # Add 10 to entire subtree of node 0
        
        print(f"   After adding 10 to subtree of node 0:")
        print(f"   Subtree sum of node 1: {tree_dp.subtree_sum(1)}")
        print(f"   Subtree sum of node 2: {tree_dp.subtree_sum(2)}")
        
        # Path update
        tree_dp.path_update(3, 6, 5)  # Add 5 to path from node 3 to node 6
        
        print(f"   After adding 5 to path 3-6:")
        print(f"   Path sum 3-6: {tree_dp.path_sum(3, 6)}")


class LazySegmentTreeNode:
    """Node in lazy propagation segment tree"""
    
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.sum = 0
        self.lazy = 0  # Lazy propagation value
        self.left: Optional['LazySegmentTreeNode'] = None
        self.right: Optional['LazySegmentTreeNode'] = None


class LazySegmentTree:
    """
    Segment tree with lazy propagation for range updates
    
    Supports range sum queries and range addition updates in O(log n)
    """
    
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.arr = arr[:]
        self.root = self._build_tree(0, self.n - 1)
        self.operations_count = 0
        self.nodes_touched = 0
    
    def _build_tree(self, start: int, end: int) -> LazySegmentTreeNode:
        """Build segment tree recursively"""
        node = LazySegmentTreeNode(start, end)
        
        if start == end:
            node.sum = self.arr[start]
        else:
            mid = (start + end) // 2
            node.left = self._build_tree(start, mid)
            node.right = self._build_tree(mid + 1, end)
            node.sum = node.left.sum + node.right.sum
        
        return node
    
    def range_update(self, left: int, right: int, delta: int) -> None:
        """Add delta to all elements in range [left, right]"""
        self.operations_count += 1
        self._range_update_recursive(self.root, left, right, delta)
    
    def _range_update_recursive(self, node: LazySegmentTreeNode, left: int, right: int, delta: int) -> None:
        """Recursive range update with lazy propagation"""
        self.nodes_touched += 1
        
        # Push down lazy value if exists
        if node.lazy != 0:
            node.sum += node.lazy * (node.end - node.start + 1)
            
            if node.start != node.end:  # Not a leaf
                if node.left:
                    node.left.lazy += node.lazy
                if node.right:
                    node.right.lazy += node.lazy
            
            node.lazy = 0
        
        # No overlap
        if node.start > right or node.end < left:
            return
        
        # Complete overlap
        if node.start >= left and node.end <= right:
            node.lazy += delta
            node.sum += delta * (node.end - node.start + 1)
            return
        
        # Partial overlap - recurse to children
        self._range_update_recursive(node.left, left, right, delta)
        self._range_update_recursive(node.right, left, right, delta)
        
        # Update current node sum
        left_sum = node.left.sum + node.left.lazy * (node.left.end - node.left.start + 1)
        right_sum = node.right.sum + node.right.lazy * (node.right.end - node.right.start + 1)
        node.sum = left_sum + right_sum
    
    def range_query(self, left: int, right: int) -> int:
        """Get sum of elements in range [left, right]"""
        self.operations_count += 1
        return self._range_query_recursive(self.root, left, right)
    
    def _range_query_recursive(self, node: LazySegmentTreeNode, left: int, right: int) -> int:
        """Recursive range query with lazy propagation"""
        self.nodes_touched += 1
        
        # No overlap
        if node.start > right or node.end < left:
            return 0
        
        # Push down lazy value if exists
        if node.lazy != 0:
            node.sum += node.lazy * (node.end - node.start + 1)
            
            if node.start != node.end:  # Not a leaf
                if node.left:
                    node.left.lazy += node.lazy
                if node.right:
                    node.right.lazy += node.lazy
            
            node.lazy = 0
        
        # Complete overlap
        if node.start >= left and node.end <= right:
            return node.sum
        
        # Partial overlap
        return (self._range_query_recursive(node.left, left, right) +
                self._range_query_recursive(node.right, left, right))


class TreeDPWithLazy:
    """
    Tree dynamic programming with lazy propagation
    
    Supports subtree and path updates/queries efficiently
    """
    
    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.values = [0] * n
        self.lazy_subtree = [0] * n  # Lazy values for subtree updates
        self.lazy_path = [0] * n     # Lazy values for path updates
        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [0] * n
    
    def add_edge(self, u: int, v: int) -> None:
        """Add edge to tree"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def set_value(self, node: int, value: int) -> None:
        """Set value at node"""
        self.values[node] = value
    
    def _preprocess(self, root: int = 0) -> None:
        """Preprocess tree for efficient queries"""
        visited = [False] * self.n
        
        def dfs(node, par, d):
            visited[node] = True
            self.parent[node] = par
            self.depth[node] = d
            self.subtree_size[node] = 1
            
            for child in self.graph[node]:
                if not visited[child]:
                    dfs(child, node, d + 1)
                    self.subtree_size[node] += self.subtree_size[child]
        
        dfs(root, -1, 0)
    
    def subtree_update(self, node: int, delta: int) -> None:
        """Add delta to all nodes in subtree of node"""
        if self.parent[0] == -1:  # Not preprocessed
            self._preprocess()
        
        self.lazy_subtree[node] += delta
    
    def subtree_sum(self, node: int) -> int:
        """Get sum of all values in subtree of node"""
        if self.parent[0] == -1:  # Not preprocessed
            self._preprocess()
        
        total = 0
        
        def dfs(current):
            nonlocal total
            # Add current node value plus all lazy updates affecting it
            node_value = self.values[current]
            
            # Add lazy updates from ancestors
            temp = current
            while temp != -1:
                node_value += self.lazy_subtree[temp]
                temp = self.parent[temp]
            
            total += node_value
            
            # Recurse to children
            for child in self.graph[current]:
                if child != self.parent[current]:
                    dfs(child)
        
        dfs(node)
        return total
    
    def path_update(self, u: int, v: int, delta: int) -> None:
        """Add delta to all nodes on path from u to v"""
        if self.parent[0] == -1:  # Not preprocessed
            self._preprocess()
        
        # Find LCA and update path
        path_u = self._get_path_to_root(u)
        path_v = self._get_path_to_root(v)
        
        # Find LCA
        lca = self._find_lca(u, v)
        
        # Update path from u to LCA
        current = u
        while current != lca:
            self.lazy_path[current] += delta
            current = self.parent[current]
        
        # Update path from v to LCA
        current = v
        while current != lca:
            self.lazy_path[current] += delta
            current = self.parent[current]
        
        # Update LCA
        self.lazy_path[lca] += delta
    
    def path_sum(self, u: int, v: int) -> int:
        """Get sum of all values on path from u to v"""
        if self.parent[0] == -1:  # Not preprocessed
            self._preprocess()
        
        lca = self._find_lca(u, v)
        total = 0
        
        # Sum from u to LCA
        current = u
        while current != lca:
            total += self.values[current] + self.lazy_path[current]
            current = self.parent[current]
        
        # Sum from v to LCA
        current = v
        while current != lca:
            total += self.values[current] + self.lazy_path[current]
            current = self.parent[current]
        
        # Add LCA
        total += self.values[lca] + self.lazy_path[lca]
        
        return total
    
    def _get_path_to_root(self, node: int) -> List[int]:
        """Get path from node to root"""
        path = []
        current = node
        while current != -1:
            path.append(current)
            current = self.parent[current]
        return path
    
    def _find_lca(self, u: int, v: int) -> int:
        """Find lowest common ancestor of u and v"""
        # Simple LCA using parent pointers
        path_u = set(self._get_path_to_root(u))
        
        current = v
        while current not in path_u:
            current = self.parent[current]
        
        return current


# ==========================================
# 3. TREE ISOMORPHISM AND CANONICALIZATION
# ==========================================

class TreeIsomorphism:
    """
    Tree isomorphism detection and canonicalization
    """
    
    def explain_tree_isomorphism(self) -> None:
        """Explain tree isomorphism concepts"""
        print("=== TREE ISOMORPHISM AND CANONICALIZATION ===")
        print("Detecting structural equivalence between trees")
        print()
        print("ISOMORPHISM CONCEPTS:")
        print("• Two trees are isomorphic if one can be transformed into the other")
        print("• Transformation: relabeling vertices preserving adjacency")
        print("• Rooted vs unrooted tree isomorphism")
        print("• Canonical form: unique representation for each isomorphism class")
        print()
        print("ALGORITHMS:")
        print("• AHU algorithm: Linear time for rooted trees")
        print("• Tree canonical form: Hash-based approaches")
        print("• Center-based canonicalization: Use tree centers")
        print("• Recursive hashing: Bottom-up hash computation")
        print()
        print("APPLICATIONS:")
        print("• Chemical compound analysis (molecular graphs)")
        print("• Compiler optimization (expression tree equivalence)")
        print("• Database query optimization")
        print("• Graph mining and pattern matching")
        print()
        print("COMPLEXITY:")
        print("• Rooted tree isomorphism: O(n) time")
        print("• Unrooted tree isomorphism: O(n) time")
        print("• General graph isomorphism: Open problem (GI-complete)")
        print("• Canonical form computation: O(n) to O(n log n)")
    
    def demonstrate_tree_isomorphism(self) -> None:
        """Demonstrate tree isomorphism algorithms"""
        print("=== TREE ISOMORPHISM DEMONSTRATIONS ===")
        
        print("1. ROOTED TREE ISOMORPHISM")
        print("   AHU algorithm for rooted tree comparison")
        
        iso_detector = RootedTreeIsomorphism()
        
        # Create two isomorphic trees
        tree1 = {
            0: [1, 2],
            1: [3, 4],
            2: [5],
            3: [],
            4: [],
            5: [6, 7],
            6: [],
            7: []
        }
        
        tree2 = {
            0: [1, 2],
            1: [3],
            2: [4, 5],
            3: [6, 7],
            4: [],
            5: [],
            6: [],
            7: []
        }
        
        # Create a non-isomorphic tree
        tree3 = {
            0: [1, 2, 3],
            1: [],
            2: [],
            3: [4, 5],
            4: [],
            5: []
        }
        
        print("   Tree structures defined")
        
        # Compute canonical forms
        canon1 = iso_detector.canonical_form(tree1, 0)
        canon2 = iso_detector.canonical_form(tree2, 0)
        canon3 = iso_detector.canonical_form(tree3, 0)
        
        print(f"   Tree 1 canonical form: {canon1}")
        print(f"   Tree 2 canonical form: {canon2}")
        print(f"   Tree 3 canonical form: {canon3}")
        
        # Check isomorphism
        iso_12 = iso_detector.are_isomorphic(tree1, 0, tree2, 0)
        iso_13 = iso_detector.are_isomorphic(tree1, 0, tree3, 0)
        
        print(f"   Tree 1 ≅ Tree 2: {iso_12}")
        print(f"   Tree 1 ≅ Tree 3: {iso_13}")
        print()
        
        print("2. UNROOTED TREE ISOMORPHISM")
        print("   Center-based approach for unrooted trees")
        
        unrooted_detector = UnrootedTreeIsomorphism()
        
        # Define trees as edge lists
        edges1 = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]
        edges2 = [(0, 1), (0, 2), (1, 3), (2, 4), (4, 5)]
        edges3 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]  # Path graph
        
        # Find centers and check isomorphism
        centers1 = unrooted_detector.find_centers(edges1)
        centers2 = unrooted_detector.find_centers(edges2)
        centers3 = unrooted_detector.find_centers(edges3)
        
        print(f"   Tree 1 centers: {centers1}")
        print(f"   Tree 2 centers: {centers2}")
        print(f"   Tree 3 centers: {centers3}")
        
        iso_unrooted = unrooted_detector.are_isomorphic(edges1, edges2)
        iso_path = unrooted_detector.are_isomorphic(edges1, edges3)
        
        print(f"   Unrooted tree 1 ≅ tree 2: {iso_unrooted}")
        print(f"   Unrooted tree 1 ≅ tree 3 (path): {iso_path}")


class RootedTreeIsomorphism:
    """
    Rooted tree isomorphism using AHU algorithm
    
    Linear time algorithm for rooted tree isomorphism
    """
    
    def canonical_form(self, tree: Dict[int, List[int]], root: int) -> str:
        """Compute canonical form using AHU algorithm"""
        
        def compute_hash(node):
            # Get hashes of all children
            child_hashes = []
            for child in tree.get(node, []):
                child_hashes.append(compute_hash(child))
            
            # Sort child hashes for canonical ordering
            child_hashes.sort()
            
            # Create canonical string
            if not child_hashes:
                return "()"
            else:
                return f"({''.join(child_hashes)})"
        
        return compute_hash(root)
    
    def are_isomorphic(self, tree1: Dict[int, List[int]], root1: int,
                      tree2: Dict[int, List[int]], root2: int) -> bool:
        """Check if two rooted trees are isomorphic"""
        canon1 = self.canonical_form(tree1, root1)
        canon2 = self.canonical_form(tree2, root2)
        return canon1 == canon2


class UnrootedTreeIsomorphism:
    """
    Unrooted tree isomorphism using center-based approach
    """
    
    def find_centers(self, edges: List[Tuple[int, int]]) -> List[int]:
        """Find centers of unrooted tree"""
        # Build adjacency list
        graph = defaultdict(list)
        degree = defaultdict(int)
        
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
            degree[u] += 1
            degree[v] += 1
        
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        
        n = len(nodes)
        
        if n == 1:
            return list(nodes)
        
        # Remove leaves iteratively
        queue = deque()
        for node in nodes:
            if degree[node] == 1:
                queue.append(node)
        
        remaining = n
        
        while remaining > 2:
            leaves_count = len(queue)
            remaining -= leaves_count
            
            for _ in range(leaves_count):
                leaf = queue.popleft()
                
                for neighbor in graph[leaf]:
                    degree[neighbor] -= 1
                    if degree[neighbor] == 1:
                        queue.append(neighbor)
        
        # Remaining nodes are centers
        centers = []
        for node in nodes:
            if degree[node] > 0:
                centers.append(node)
        
        return centers
    
    def are_isomorphic(self, edges1: List[Tuple[int, int]], 
                      edges2: List[Tuple[int, int]]) -> bool:
        """Check if two unrooted trees are isomorphic"""
        # Check basic properties first
        if len(edges1) != len(edges2):
            return False
        
        # Find centers
        centers1 = self.find_centers(edges1)
        centers2 = self.find_centers(edges2)
        
        if len(centers1) != len(centers2):
            return False
        
        # Convert to rooted trees and compare
        rooted_iso = RootedTreeIsomorphism()
        
        def edges_to_tree(edges, root):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            # Convert to rooted tree
            tree = {}
            visited = set()
            
            def dfs(node, parent):
                visited.add(node)
                tree[node] = []
                
                for neighbor in graph[node]:
                    if neighbor != parent and neighbor not in visited:
                        tree[node].append(neighbor)
                        dfs(neighbor, node)
            
            dfs(root, -1)
            return tree
        
        # Try all combinations of centers
        for center1 in centers1:
            tree1 = edges_to_tree(edges1, center1)
            
            for center2 in centers2:
                tree2 = edges_to_tree(edges2, center2)
                
                if rooted_iso.are_isomorphic(tree1, center1, tree2, center2):
                    return True
        
        return False


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_advanced_tree_techniques():
    """Demonstrate all advanced tree techniques"""
    print("=== ADVANCED TREE TECHNIQUES COMPREHENSIVE GUIDE ===\n")
    
    techniques = AdvancedTreeTechniques()
    
    # 1. Persistent trees
    techniques.explain_persistent_trees()
    print("\n" + "="*60 + "\n")
    
    techniques.demonstrate_persistent_trees()
    print("\n" + "="*60 + "\n")
    
    # 2. Lazy propagation
    lazy_trees = LazyPropagationTrees()
    lazy_trees.explain_lazy_propagation()
    print("\n" + "="*60 + "\n")
    
    lazy_trees.demonstrate_lazy_propagation()
    print("\n" + "="*60 + "\n")
    
    # 3. Tree isomorphism
    isomorphism = TreeIsomorphism()
    isomorphism.explain_tree_isomorphism()
    print("\n" + "="*60 + "\n")
    
    isomorphism.demonstrate_tree_isomorphism()


if __name__ == "__main__":
    demonstrate_advanced_tree_techniques()
    
    print("\n=== ADVANCED TREE TECHNIQUES MASTERY GUIDE ===")
    
    print("\n🎯 ADVANCED TECHNIQUE CATEGORIES:")
    print("• Persistent Structures: Immutable trees with version control")
    print("• Lazy Propagation: Deferred updates for range operations")
    print("• Tree Isomorphism: Structural equivalence detection")
    print("• Advanced Algorithms: Research-level tree techniques")
    
    print("\n📊 COMPLEXITY ANALYSIS:")
    print("• Persistent operations: O(log n) time, O(n log n) space for versions")
    print("• Lazy propagation: O(log n) amortized for range updates")
    print("• Tree isomorphism: O(n) time for trees")
    print("• Memory efficiency varies by technique and implementation")
    
    print("\n⚡ RESEARCH APPLICATIONS:")
    print("• Functional programming languages: Persistent data structures")
    print("• Database systems: Versioning and time-travel queries")
    print("• Bioinformatics: Tree comparison and phylogenetic analysis")
    print("• Compiler optimization: Expression tree equivalence")
    print("• Version control systems: Efficient diff and merge operations")
    
    print("\n🔧 IMPLEMENTATION STRATEGIES:")
    print("• Study theoretical foundations thoroughly")
    print("• Implement simple versions before optimizing")
    print("• Use appropriate data structures for each technique")
    print("• Consider memory vs time trade-offs carefully")
    print("• Profile and benchmark against existing implementations")
    
    print("\n🏆 CUTTING-EDGE TOPICS:")
    print("• Cache-oblivious persistent structures")
    print("• Concurrent persistent data structures")
    print("• Succinct tree representations")
    print("• Dynamic tree decomposition algorithms")
    print("• Machine learning on tree structures")
    
    print("\n🎓 RESEARCH ROADMAP:")
    print("1. Master fundamental tree algorithms")
    print("2. Study functional programming paradigms")
    print("3. Learn advanced algorithmic techniques")
    print("4. Read current research papers")
    print("5. Implement research prototypes")
    print("6. Contribute to open-source projects")
    
    print("\n💡 RESEARCH SUCCESS TIPS:")
    print("• Read papers from top-tier conferences (STOC, FOCS, SODA)")
    print("• Implement algorithms from scratch to understand deeply")
    print("• Collaborate with researchers in related fields")
    print("• Stay updated with latest developments")
    print("• Focus on both theoretical analysis and practical implementation")
    print("• Consider interdisciplinary applications")
