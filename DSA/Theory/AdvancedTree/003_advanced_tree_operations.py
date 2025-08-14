"""
Advanced Tree Operations and Algorithms
=======================================

Topics: Tree traversals, LCA, tree decomposition, heavy-light decomposition
Companies: Google, Amazon, Microsoft, Facebook, Apple, competitive programming
Difficulty: Hard to Expert
Time Complexity: Various (O(log n) to O(n) depending on operation)
Space Complexity: O(n) for tree storage, O(log n) to O(n) for algorithms
"""

from typing import List, Optional, Dict, Any, Tuple, Set, Callable
from collections import defaultdict, deque
import math

class AdvancedTreeOperations:
    
    def __init__(self):
        """Initialize with advanced operations tracking"""
        self.operation_count = 0
        self.algorithm_stats = {}
    
    # ==========================================
    # 1. ADVANCED TRAVERSAL ALGORITHMS
    # ==========================================
    
    def explain_advanced_traversals(self) -> None:
        """
        Explain advanced tree traversal techniques beyond basic DFS/BFS
        """
        print("=== ADVANCED TREE TRAVERSAL ALGORITHMS ===")
        print("Beyond basic DFS/BFS - specialized traversal techniques")
        print()
        print("MORRIS TRAVERSAL:")
        print("• In-order traversal without recursion or stack")
        print("• Uses threaded binary tree concept")
        print("• O(n) time, O(1) space complexity")
        print("• Temporarily modifies tree structure")
        print()
        print("VERTICAL ORDER TRAVERSAL:")
        print("• Process nodes by their horizontal distance from root")
        print("• Uses coordinate system (x, y)")
        print("• Important for tree visualization")
        print("• Applications: GUI layout, 2D representations")
        print()
        print("BOUNDARY TRAVERSAL:")
        print("• Traverse only the boundary nodes of tree")
        print("• Left boundary + leaves + right boundary")
        print("• Used in tree printing and visualization")
        print("• Applications: Tree outline, perimeter calculation")
        print()
        print("DIAGONAL TRAVERSAL:")
        print("• Process nodes along diagonal lines")
        print("• Nodes at same diagonal have equal (row - col)")
        print("• Useful for certain tree problems")
        print("• Applications: Matrix representations")
        print()
        print("EULER TOUR:")
        print("• Visit each edge twice (down and up)")
        print("• Foundation for many advanced algorithms")
        print("• Enables LCA preprocessing and range queries")
        print("• Applications: LCA, subtree queries, rerooting")
    
    def demonstrate_morris_traversal(self) -> None:
        """
        Demonstrate Morris in-order traversal (O(1) space)
        """
        print("=== MORRIS TRAVERSAL DEMONSTRATION ===")
        print("In-order traversal with O(1) space complexity")
        print()
        
        # Create sample tree
        tree = TreeNode(4)
        tree.left = TreeNode(2)
        tree.right = TreeNode(6)
        tree.left.left = TreeNode(1)
        tree.left.right = TreeNode(3)
        tree.right.left = TreeNode(5)
        tree.right.right = TreeNode(7)
        
        print("Sample tree structure:")
        print("       4")
        print("      / \\")
        print("     2   6")
        print("    / \\ / \\")
        print("   1  3 5  7")
        print()
        
        morris = MorrisTraversal()
        result = morris.inorder_morris(tree)
        
        print(f"Morris in-order result: {result}")
        print("Expected: [1, 2, 3, 4, 5, 6, 7]")
        print()
    
    def demonstrate_vertical_traversal(self) -> None:
        """
        Demonstrate vertical order traversal
        """
        print("=== VERTICAL ORDER TRAVERSAL ===")
        print("Process nodes by horizontal distance from root")
        print()
        
        # Create sample tree
        tree = TreeNode(1)
        tree.left = TreeNode(2)
        tree.right = TreeNode(3)
        tree.left.left = TreeNode(4)
        tree.left.right = TreeNode(5)
        tree.right.left = TreeNode(6)
        tree.right.right = TreeNode(7)
        
        print("Sample tree with coordinates:")
        print("       1 (0,0)")
        print("      /   \\")
        print("   2(-1,1) 3(1,1)")
        print("   /  \\    /  \\")
        print("4(-2,2)5(0,2)6(0,2)7(2,2)")
        print()
        
        vertical = VerticalTraversal()
        result = vertical.vertical_order(tree)
        
        print("Vertical order result:")
        for i, column in enumerate(result):
            print(f"  Column {i}: {column}")


class TreeNode:
    """Standard binary tree node"""
    
    def __init__(self, val: int = 0):
        self.val = val
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None


class MorrisTraversal:
    """
    Morris traversal implementation for O(1) space in-order traversal
    
    Uses threaded binary tree concept to eliminate recursion stack
    """
    
    def inorder_morris(self, root: Optional[TreeNode]) -> List[int]:
        """
        Morris in-order traversal
        
        Time: O(n), Space: O(1)
        """
        result = []
        current = root
        
        print("Morris traversal steps:")
        step = 1
        
        while current:
            print(f"Step {step}: At node {current.val if current else 'None'}")
            
            if not current.left:
                # No left subtree, visit current and go right
                print(f"  No left child, visit {current.val}")
                result.append(current.val)
                current = current.right
            else:
                # Find inorder predecessor
                predecessor = current.left
                
                # Go to rightmost node in left subtree
                while predecessor.right and predecessor.right != current:
                    predecessor = predecessor.right
                
                if not predecessor.right:
                    # Make current the right child of predecessor
                    print(f"  Create thread: {predecessor.val} -> {current.val}")
                    predecessor.right = current
                    current = current.left
                else:
                    # Thread already exists, remove it
                    print(f"  Remove thread: {predecessor.val} -X-> {current.val}")
                    predecessor.right = None
                    print(f"  Visit {current.val}")
                    result.append(current.val)
                    current = current.right
            
            step += 1
        
        return result


class VerticalTraversal:
    """
    Vertical order traversal implementation
    
    Groups nodes by their horizontal distance from root
    """
    
    def vertical_order(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Vertical order traversal
        
        Time: O(n log n), Space: O(n)
        """
        if not root:
            return []
        
        # Map from column to list of (row, value)
        column_map = defaultdict(list)
        
        # BFS with coordinates
        queue = deque([(root, 0, 0)])  # (node, row, col)
        
        print("BFS traversal with coordinates:")
        
        while queue:
            node, row, col = queue.popleft()
            print(f"  Node {node.val} at ({row}, {col})")
            
            column_map[col].append((row, node.val))
            
            if node.left:
                queue.append((node.left, row + 1, col - 1))
            if node.right:
                queue.append((node.right, row + 1, col + 1))
        
        # Sort columns and within each column sort by row
        result = []
        for col in sorted(column_map.keys()):
            column_nodes = sorted(column_map[col])
            result.append([val for row, val in column_nodes])
        
        return result


# ==========================================
# 2. LOWEST COMMON ANCESTOR (LCA) ALGORITHMS
# ==========================================

class LCAAlgorithms:
    """
    Multiple algorithms for Lowest Common Ancestor queries
    
    Includes naive, preprocessing-based, and advanced techniques
    """
    
    def explain_lca_techniques(self) -> None:
        """Explain different LCA algorithms and their trade-offs"""
        print("=== LOWEST COMMON ANCESTOR ALGORITHMS ===")
        print("Finding LCA of two nodes in a tree")
        print()
        print("ALGORITHM COMPARISON:")
        print("┌─────────────────────┬─────────────┬─────────────┬─────────────┐")
        print("│ Algorithm           │ Preprocess  │ Query Time  │ Space       │")
        print("├─────────────────────┼─────────────┼─────────────┼─────────────┤")
        print("│ Naive (DFS paths)   │ O(1)        │ O(n)        │ O(1)        │")
        print("│ Parent pointers     │ O(n)        │ O(h)        │ O(n)        │")
        print("│ Binary lifting      │ O(n log n)  │ O(log n)    │ O(n log n)  │")
        print("│ Euler tour + RMQ    │ O(n log n)  │ O(log n)    │ O(n log n)  │")
        print("│ Tarjan's offline    │ O(n α(n))   │ O(1)*       │ O(n)        │")
        print("└─────────────────────┴─────────────┴─────────────┴─────────────┘")
        print("* Amortized over all queries")
        print()
        print("WHEN TO USE EACH:")
        print("• Naive: Single queries, small trees")
        print("• Parent pointers: Few queries, balanced trees")
        print("• Binary lifting: Many queries, general trees")
        print("• Euler tour: Online queries, complex operations")
        print("• Tarjan's: Offline queries, optimal performance")
    
    def demonstrate_binary_lifting_lca(self) -> None:
        """
        Demonstrate binary lifting LCA algorithm
        """
        print("=== BINARY LIFTING LCA DEMONSTRATION ===")
        print("Efficient LCA queries using binary lifting preprocessing")
        print()
        
        # Create sample tree
        tree = self._create_sample_tree()
        
        print("Sample tree structure:")
        print("       1")
        print("      / \\")
        print("     2   3")
        print("    /|   |\\")
        print("   4 5   6 7")
        print("     |     |")
        print("     8     9")
        print()
        
        lca_solver = BinaryLiftingLCA(tree, 10)  # max 10 nodes
        
        # Test LCA queries
        test_queries = [(8, 9), (4, 5), (8, 6), (2, 3)]
        
        print("LCA queries:")
        for u, v in test_queries:
            lca = lca_solver.query_lca(u, v)
            print(f"  LCA({u}, {v}) = {lca}")
    
    def _create_sample_tree(self) -> Dict[int, List[int]]:
        """Create sample tree for LCA demonstration"""
        tree = defaultdict(list)
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (5, 8), (7, 9)]
        
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)
        
        return tree


class BinaryLiftingLCA:
    """
    Binary lifting LCA implementation
    
    Preprocessing: O(n log n)
    Query: O(log n)
    Space: O(n log n)
    """
    
    def __init__(self, tree: Dict[int, List[int]], n: int, root: int = 1):
        self.tree = tree
        self.n = n
        self.root = root
        self.LOG = 20  # log2(max_n)
        
        # Binary lifting table: parent[i][v] = 2^i-th ancestor of v
        self.parent = [[-1] * (n + 1) for _ in range(self.LOG)]
        self.depth = [0] * (n + 1)
        
        self._preprocess()
    
    def _preprocess(self) -> None:
        """Preprocess the tree for LCA queries"""
        print("Preprocessing tree for binary lifting:")
        
        # DFS to compute depths and direct parents
        visited = set()
        
        def dfs(v, d, p):
            visited.add(v)
            self.depth[v] = d
            self.parent[0][v] = p
            
            for u in self.tree[v]:
                if u not in visited:
                    dfs(u, d + 1, v)
        
        dfs(self.root, 0, -1)
        
        print(f"  Computed depths: {dict(enumerate(self.depth))}")
        
        # Fill binary lifting table
        for i in range(1, self.LOG):
            for v in range(1, self.n + 1):
                if self.parent[i-1][v] != -1:
                    self.parent[i][v] = self.parent[i-1][self.parent[i-1][v]]
        
        print(f"  Built binary lifting table (2^0 to 2^{self.LOG-1} ancestors)")
    
    def query_lca(self, u: int, v: int) -> int:
        """
        Query LCA of two nodes
        
        Time: O(log n)
        """
        print(f"    Finding LCA of {u} and {v}:")
        
        # Make u the deeper node
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        
        print(f"      Node depths: {u} -> {self.depth[u]}, {v} -> {self.depth[v]}")
        
        # Bring u to the same level as v
        diff = self.depth[u] - self.depth[v]
        
        for i in range(self.LOG):
            if (diff >> i) & 1:
                print(f"      Lifting {u} by 2^{i} = {1 << i}")
                u = self.parent[i][u]
        
        print(f"      After leveling: {u} and {v} at depth {self.depth[v]}")
        
        if u == v:
            return u
        
        # Binary search for LCA
        for i in range(self.LOG - 1, -1, -1):
            if (self.parent[i][u] != self.parent[i][v] and 
                self.parent[i][u] != -1):
                print(f"      Binary lifting: 2^{i} step")
                u = self.parent[i][u]
                v = self.parent[i][v]
        
        lca = self.parent[0][u]
        print(f"      Found LCA: {lca}")
        return lca


# ==========================================
# 3. TREE DECOMPOSITION TECHNIQUES
# ==========================================

class TreeDecomposition:
    """
    Tree decomposition techniques for advanced tree algorithms
    
    Includes centroid decomposition and heavy-light decomposition
    """
    
    def explain_decomposition_techniques(self) -> None:
        """Explain tree decomposition techniques"""
        print("=== TREE DECOMPOSITION TECHNIQUES ===")
        print("Advanced techniques for efficient tree queries")
        print()
        print("CENTROID DECOMPOSITION:")
        print("• Recursively remove centroid to create decomposition tree")
        print("• Centroid: node whose removal creates subtrees of size ≤ n/2")
        print("• Creates decomposition tree of height O(log n)")
        print("• Applications: Path queries, distance queries")
        print()
        print("HEAVY-LIGHT DECOMPOSITION:")
        print("• Decompose tree into heavy and light edges")
        print("• Heavy edge: connects node to its largest subtree")
        print("• Creates O(log n) light edges on any root-to-leaf path")
        print("• Applications: Path updates, subtree queries")
        print()
        print("LINK-CUT TREES:")
        print("• Dynamic tree data structure")
        print("• Supports link, cut, and path queries")
        print("• Based on splay trees and heavy-light decomposition")
        print("• Applications: Dynamic connectivity, network flows")
        print()
        print("APPLICATIONS:")
        print("• Path queries in logarithmic time")
        print("• Subtree updates and queries")
        print("• Distance queries between any two nodes")
        print("• Dynamic tree connectivity problems")
    
    def demonstrate_centroid_decomposition(self) -> None:
        """
        Demonstrate centroid decomposition
        """
        print("=== CENTROID DECOMPOSITION DEMONSTRATION ===")
        print("Decomposing tree using centroid removal")
        print()
        
        # Create sample tree
        tree = defaultdict(list)
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (4, 7), (4, 8)]
        
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)
        
        print("Original tree structure:")
        print("     1")
        print("    / \\")
        print("   2   3")
        print("  /|   |")
        print(" 4 5   6")
        print("/|")
        print("7 8")
        print()
        
        centroid_decomp = CentroidDecomposition(tree, 8)
        decomp_tree = centroid_decomp.build_decomposition()
        
        print("Centroid decomposition tree:")
        centroid_decomp.display_decomposition(decomp_tree)
    
    def demonstrate_heavy_light_decomposition(self) -> None:
        """
        Demonstrate heavy-light decomposition
        """
        print("=== HEAVY-LIGHT DECOMPOSITION DEMONSTRATION ===")
        print("Decomposing tree into heavy and light edges")
        print()
        
        # Create sample tree
        tree = defaultdict(list)
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (5, 7), (5, 8)]
        
        for u, v in edges:
            tree[u].append(v)
            tree[v].append(u)
        
        print("Original tree structure:")
        print("     1")
        print("    / \\")
        print("   2   3")
        print("  /|   |")
        print(" 4 5   6")
        print("  /|")
        print(" 7 8")
        print()
        
        hld = HeavyLightDecomposition(tree, 8)
        hld.decompose()
        
        print("Heavy-light decomposition result:")
        hld.display_chains()


class CentroidDecomposition:
    """
    Centroid decomposition implementation
    
    Creates decomposition tree of height O(log n)
    """
    
    def __init__(self, tree: Dict[int, List[int]], n: int):
        self.tree = tree
        self.n = n
        self.removed = set()
        self.subtree_size = {}
    
    def build_decomposition(self) -> Dict[int, List[int]]:
        """Build centroid decomposition tree"""
        decomp_tree = defaultdict(list)
        
        def decompose(component_nodes, parent_centroid=None):
            if not component_nodes:
                return
            
            # Find centroid of current component
            centroid = self._find_centroid(component_nodes)
            
            print(f"  Found centroid: {centroid} for component {sorted(component_nodes)}")
            
            if parent_centroid is not None:
                decomp_tree[parent_centroid].append(centroid)
            
            # Mark centroid as removed
            self.removed.add(centroid)
            
            # Find connected components after removing centroid
            components = self._get_components_after_removal(centroid, component_nodes)
            
            # Recursively decompose each component
            for component in components:
                decompose(component, centroid)
        
        all_nodes = set(range(1, self.n + 1))
        decompose(all_nodes)
        
        return decomp_tree
    
    def _find_centroid(self, component_nodes: Set[int]) -> int:
        """Find centroid of component"""
        component_size = len(component_nodes)
        
        # Pick any node as root for this component
        root = next(iter(component_nodes))
        
        # Calculate subtree sizes
        self._calculate_subtree_sizes(root, -1, component_nodes)
        
        # Find centroid
        return self._find_centroid_dfs(root, -1, component_size, component_nodes)
    
    def _calculate_subtree_sizes(self, v: int, parent: int, component: Set[int]) -> int:
        """Calculate subtree sizes within component"""
        size = 1
        
        for u in self.tree[v]:
            if u != parent and u in component and u not in self.removed:
                size += self._calculate_subtree_sizes(u, v, component)
        
        self.subtree_size[v] = size
        return size
    
    def _find_centroid_dfs(self, v: int, parent: int, component_size: int, component: Set[int]) -> int:
        """Find centroid using DFS"""
        for u in self.tree[v]:
            if (u != parent and u in component and u not in self.removed and
                self.subtree_size[u] > component_size // 2):
                return self._find_centroid_dfs(u, v, component_size, component)
        
        return v
    
    def _get_components_after_removal(self, centroid: int, original_component: Set[int]) -> List[Set[int]]:
        """Get connected components after removing centroid"""
        remaining_nodes = original_component - {centroid} - self.removed
        components = []
        visited = set()
        
        for neighbor in self.tree[centroid]:
            if neighbor in remaining_nodes and neighbor not in visited:
                component = set()
                self._dfs_component(neighbor, component, remaining_nodes, visited)
                components.append(component)
        
        return components
    
    def _dfs_component(self, v: int, component: Set[int], remaining: Set[int], visited: Set[int]):
        """DFS to find connected component"""
        visited.add(v)
        component.add(v)
        
        for u in self.tree[v]:
            if u in remaining and u not in visited:
                self._dfs_component(u, component, remaining, visited)
    
    def display_decomposition(self, decomp_tree: Dict[int, List[int]]):
        """Display centroid decomposition tree"""
        def dfs_display(v, depth=0):
            print("  " + "  " * depth + f"Centroid {v}")
            for child in decomp_tree[v]:
                dfs_display(child, depth + 1)
        
        # Find root of decomposition tree
        all_nodes = set()
        children = set()
        
        for parent, child_list in decomp_tree.items():
            all_nodes.add(parent)
            children.update(child_list)
        
        roots = all_nodes - children
        
        for root in roots:
            dfs_display(root)


class HeavyLightDecomposition:
    """
    Heavy-light decomposition implementation
    
    Decomposes tree into heavy and light edges for efficient path queries
    """
    
    def __init__(self, tree: Dict[int, List[int]], n: int, root: int = 1):
        self.tree = tree
        self.n = n
        self.root = root
        
        # HLD arrays
        self.parent = [-1] * (n + 1)
        self.depth = [0] * (n + 1)
        self.subtree_size = [0] * (n + 1)
        self.heavy_child = [-1] * (n + 1)
        self.chain_head = [-1] * (n + 1)
        self.chains = []
    
    def decompose(self):
        """Perform heavy-light decomposition"""
        print("Step 1: Computing subtree sizes and heavy children")
        self._compute_heavy_children(self.root, -1, 0)
        
        print("Step 2: Building heavy-light chains")
        self._build_chains()
    
    def _compute_heavy_children(self, v: int, p: int, d: int):
        """Compute subtree sizes and identify heavy children"""
        self.parent[v] = p
        self.depth[v] = d
        self.subtree_size[v] = 1
        
        max_child_size = 0
        
        for u in self.tree[v]:
            if u != p:
                self._compute_heavy_children(u, v, d + 1)
                self.subtree_size[v] += self.subtree_size[u]
                
                if self.subtree_size[u] > max_child_size:
                    max_child_size = self.subtree_size[u]
                    self.heavy_child[v] = u
        
        if self.heavy_child[v] != -1:
            print(f"    Node {v}: heavy child = {self.heavy_child[v]} (size {max_child_size})")
    
    def _build_chains(self):
        """Build heavy-light chains"""
        visited = [False] * (self.n + 1)
        
        def dfs_chains(v, chain_start):
            chain = []
            current = v
            
            while current != -1 and not visited[current]:
                visited[current] = True
                chain.append(current)
                self.chain_head[current] = chain_start
                current = self.heavy_child[current]
            
            if chain:
                self.chains.append(chain)
                print(f"    Heavy chain starting at {chain_start}: {chain}")
            
            # Process light children
            for u in self.tree[v]:
                if not visited[u]:
                    dfs_chains(u, u)  # Light child starts its own chain
        
        dfs_chains(self.root, self.root)
    
    def display_chains(self):
        """Display all heavy-light chains"""
        print("Heavy-light chains:")
        for i, chain in enumerate(self.chains):
            chain_type = "Heavy" if len(chain) > 1 else "Light"
            print(f"  Chain {i} ({chain_type}): {chain}")
        
        print("\nChain heads:")
        for v in range(1, self.n + 1):
            print(f"  Node {v}: chain head = {self.chain_head[v]}")


# ==========================================
# DEMONSTRATION AND TESTING
# ==========================================

def demonstrate_advanced_tree_operations():
    """Demonstrate all advanced tree operations"""
    print("=== ADVANCED TREE OPERATIONS DEMONSTRATION ===\n")
    
    advanced_ops = AdvancedTreeOperations()
    
    # 1. Advanced traversals
    advanced_ops.explain_advanced_traversals()
    print("\n" + "="*60 + "\n")
    
    advanced_ops.demonstrate_morris_traversal()
    print("\n" + "-"*40 + "\n")
    
    advanced_ops.demonstrate_vertical_traversal()
    print("\n" + "="*60 + "\n")
    
    # 2. LCA algorithms
    lca_algos = LCAAlgorithms()
    lca_algos.explain_lca_techniques()
    print("\n" + "="*60 + "\n")
    
    lca_algos.demonstrate_binary_lifting_lca()
    print("\n" + "="*60 + "\n")
    
    # 3. Tree decomposition
    tree_decomp = TreeDecomposition()
    tree_decomp.explain_decomposition_techniques()
    print("\n" + "="*60 + "\n")
    
    tree_decomp.demonstrate_centroid_decomposition()
    print("\n" + "-"*40 + "\n")
    
    tree_decomp.demonstrate_heavy_light_decomposition()


if __name__ == "__main__":
    demonstrate_advanced_tree_operations()
    
    print("\n=== ADVANCED TREE OPERATIONS MASTERY GUIDE ===")
    
    print("\n🎯 KEY ALGORITHMS TO MASTER:")
    print("• Morris traversal for O(1) space in-order traversal")
    print("• Binary lifting LCA for efficient ancestor queries")
    print("• Centroid decomposition for distance queries")
    print("• Heavy-light decomposition for path queries")
    print("• Euler tour technique for range queries on trees")
    
    print("\n📊 COMPLEXITY COMPARISON:")
    print("• Morris traversal: O(n) time, O(1) space")
    print("• Binary lifting LCA: O(n log n) preprocess, O(log n) query")
    print("• Centroid decomposition: O(n log n) build, O(log n) query levels")
    print("• Heavy-light decomposition: O(n) build, O(log² n) path queries")
    print("• Euler tour: O(n) build, enables O(log n) range queries")
    
    print("\n⚡ OPTIMIZATION STRATEGIES:")
    print("• Choose appropriate LCA algorithm based on query frequency")
    print("• Use centroid decomposition for distance-based queries")
    print("• Apply heavy-light decomposition for path update problems")
    print("• Combine with segment trees for advanced range queries")
    print("• Consider memory vs query time trade-offs")
    
    print("\n🔧 IMPLEMENTATION TIPS:")
    print("• Practice Morris traversal threading technique")
    print("• Master binary lifting table construction")
    print("• Understand centroid properties and finding algorithms")
    print("• Learn heavy edge identification and chain building")
    print("• Implement proper tree rerooting for flexible queries")
    
    print("\n🏆 COMPETITIVE PROGRAMMING APPLICATIONS:")
    print("• LCA queries in tree problems")
    print("• Path queries and updates on trees")
    print("• Distance queries between tree nodes")
    print("• Subtree operations and modifications")
    print("• Dynamic tree connectivity problems")
    
    print("\n🎓 LEARNING PROGRESSION:")
    print("1. Master basic tree traversals and properties")
    print("2. Learn LCA algorithms from naive to advanced")
    print("3. Understand tree decomposition motivations")
    print("4. Implement centroid decomposition")
    print("5. Study heavy-light decomposition applications")
    print("6. Practice with competitive programming problems")
    
    print("\n💡 SUCCESS TIPS:")
    print("• Visualize tree decompositions by hand")
    print("• Practice implementing from scratch")
    print("• Understand the theoretical foundations")
    print("• Solve related competitive programming problems")
    print("• Study real-world applications in databases and networks")
