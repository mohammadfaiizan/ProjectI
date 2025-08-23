"""
Heavy-Light Decomposition - Advanced Tree Algorithm
Difficulty: Medium

Heavy-Light Decomposition is a technique to decompose a tree into a set of 
disjoint paths such that any path from root to any node uses at most O(log n) 
of these paths. This enables efficient path queries and updates.

Key Concepts:
1. Heavy and Light Edges
2. Path Decomposition
3. Segment Tree Integration
4. Path Queries and Updates
5. LCA using HLD
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.children = []

class HeavyLightDecomposition:
    """Heavy-Light Decomposition implementation"""
    
    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.parent = [-1] * n
        self.depth = [0] * n
        self.subtree_size = [0] * n
        self.heavy_child = [-1] * n
        self.chain_head = [-1] * n
        self.chain_pos = [-1] * n
        self.values = [0] * n
        self.pos = 0
    
    def add_edge(self, u: int, v: int):
        """Add edge to tree"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def dfs1(self, node: int, par: int, d: int):
        """
        First DFS: Calculate subtree sizes and find heavy children
        
        Time: O(n), Space: O(h)
        """
        self.parent[node] = par
        self.depth[d] = d
        self.subtree_size[node] = 1
        
        max_child_size = 0
        
        for child in self.graph[node]:
            if child != par:
                self.dfs1(child, node, d + 1)
                self.subtree_size[node] += self.subtree_size[child]
                
                if self.subtree_size[child] > max_child_size:
                    max_child_size = self.subtree_size[child]
                    self.heavy_child[node] = child
    
    def dfs2(self, node: int, par: int, head: int):
        """
        Second DFS: Decompose tree into heavy paths
        
        Time: O(n), Space: O(h)
        """
        self.chain_head[node] = head
        self.chain_pos[node] = self.pos
        self.pos += 1
        
        # Process heavy child first
        if self.heavy_child[node] != -1:
            self.dfs2(self.heavy_child[node], node, head)
        
        # Process light children
        for child in self.graph[node]:
            if child != par and child != self.heavy_child[node]:
                self.dfs2(child, node, child)
    
    def build(self, root: int = 0):
        """
        Build Heavy-Light Decomposition
        
        Time: O(n), Space: O(n)
        """
        self.dfs1(root, -1, 0)
        self.dfs2(root, -1, root)
    
    def lca(self, u: int, v: int) -> int:
        """
        Find LCA using Heavy-Light Decomposition
        
        Time: O(log n), Space: O(1)
        """
        while self.chain_head[u] != self.chain_head[v]:
            if self.depth[self.chain_head[u]] < self.depth[self.chain_head[v]]:
                u, v = v, u
            u = self.parent[self.chain_head[u]]
        
        return u if self.depth[u] < self.depth[v] else v
    
    def path_query(self, u: int, v: int) -> List[Tuple[int, int]]:
        """
        Get path segments from u to v
        
        Time: O(log n), Space: O(log n)
        """
        segments = []
        lca_node = self.lca(u, v)
        
        # Path from u to LCA
        curr = u
        while self.chain_head[curr] != self.chain_head[lca_node]:
            segments.append((self.chain_pos[self.chain_head[curr]], self.chain_pos[curr]))
            curr = self.parent[self.chain_head[curr]]
        
        if curr != lca_node:
            segments.append((self.chain_pos[lca_node], self.chain_pos[curr]))
        
        # Path from LCA to v
        path_to_v = []
        curr = v
        while self.chain_head[curr] != self.chain_head[lca_node]:
            path_to_v.append((self.chain_pos[self.chain_head[curr]], self.chain_pos[curr]))
            curr = self.parent[self.chain_head[curr]]
        
        if curr != lca_node:
            path_to_v.append((self.chain_pos[lca_node] + 1, self.chain_pos[curr]))
        
        # Reverse path to v and add to segments
        segments.extend(reversed(path_to_v))
        
        return segments

class HLDWithSegmentTree:
    """Heavy-Light Decomposition with Segment Tree for range queries"""
    
    def __init__(self, n: int):
        self.hld = HeavyLightDecomposition(n)
        self.seg_tree = [0] * (4 * n)
        self.lazy = [0] * (4 * n)
    
    def build_segment_tree(self, node: int, start: int, end: int, values: List[int]):
        """Build segment tree on HLD positions"""
        if start == end:
            self.seg_tree[node] = values[start]
        else:
            mid = (start + end) // 2
            self.build_segment_tree(2 * node, start, mid, values)
            self.build_segment_tree(2 * node + 1, mid + 1, end, values)
            self.seg_tree[node] = self.seg_tree[2 * node] + self.seg_tree[2 * node + 1]
    
    def update_range(self, node: int, start: int, end: int, l: int, r: int, val: int):
        """Range update with lazy propagation"""
        if self.lazy[node] != 0:
            self.seg_tree[node] += (end - start + 1) * self.lazy[node]
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0
        
        if start > r or end < l:
            return
        
        if start >= l and end <= r:
            self.seg_tree[node] += (end - start + 1) * val
            if start != end:
                self.lazy[2 * node] += val
                self.lazy[2 * node + 1] += val
            return
        
        mid = (start + end) // 2
        self.update_range(2 * node, start, mid, l, r, val)
        self.update_range(2 * node + 1, mid + 1, end, l, r, val)
        self.seg_tree[node] = self.seg_tree[2 * node] + self.seg_tree[2 * node + 1]
    
    def query_range(self, node: int, start: int, end: int, l: int, r: int) -> int:
        """Range sum query"""
        if start > r or end < l:
            return 0
        
        if self.lazy[node] != 0:
            self.seg_tree[node] += (end - start + 1) * self.lazy[node]
            if start != end:
                self.lazy[2 * node] += self.lazy[node]
                self.lazy[2 * node + 1] += self.lazy[node]
            self.lazy[node] = 0
        
        if start >= l and end <= r:
            return self.seg_tree[node]
        
        mid = (start + end) // 2
        return (self.query_range(2 * node, start, mid, l, r) +
                self.query_range(2 * node + 1, mid + 1, end, l, r))
    
    def path_update(self, u: int, v: int, val: int):
        """Update all nodes on path from u to v"""
        segments = self.hld.path_query(u, v)
        for l, r in segments:
            self.update_range(1, 0, self.hld.n - 1, l, r, val)
    
    def path_query_sum(self, u: int, v: int) -> int:
        """Query sum of all nodes on path from u to v"""
        segments = self.hld.path_query(u, v)
        total = 0
        for l, r in segments:
            total += self.query_range(1, 0, self.hld.n - 1, l, r)
        return total

def test_heavy_light_decomposition():
    """Test Heavy-Light Decomposition"""
    print("=== Testing Heavy-Light Decomposition ===")
    
    # Create test tree
    n = 7
    hld = HeavyLightDecomposition(n)
    
    # Add edges to form tree
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for u, v in edges:
        hld.add_edge(u, v)
    
    # Build HLD
    hld.build(0)
    
    print("Tree structure:")
    print("    0")
    print("   / \\")
    print("  1   2")
    print(" / \\ / \\")
    print("3  4 5  6")
    
    print(f"\nHLD Results:")
    print(f"Chain heads: {hld.chain_head}")
    print(f"Chain positions: {hld.chain_pos}")
    print(f"Heavy children: {hld.heavy_child}")
    
    # Test LCA queries
    test_pairs = [(3, 4), (3, 5), (4, 6), (1, 2)]
    print(f"\nLCA Queries:")
    for u, v in test_pairs:
        lca_result = hld.lca(u, v)
        print(f"LCA({u}, {v}) = {lca_result}")

def demonstrate_hld_applications():
    """Demonstrate HLD applications"""
    print("\n=== HLD Applications ===")
    
    print("Key Applications:")
    print("1. Path Queries: Sum, max, min on tree paths")
    print("2. Path Updates: Add value to all nodes on path")
    print("3. LCA Queries: O(log n) lowest common ancestor")
    print("4. Subtree Queries: Operations on entire subtrees")
    print("5. Tree Rerooting: Dynamic root changes")
    
    print("\nComplexity Benefits:")
    print("• Preprocessing: O(n)")
    print("• Path queries: O(log² n) with segment tree")
    print("• Path updates: O(log² n) with segment tree")
    print("• LCA queries: O(log n)")
    print("• Space: O(n)")
    
    print("\nReal-world Uses:")
    print("• Network analysis and routing")
    print("• Hierarchical data processing")
    print("• Game tree analysis")
    print("• Organizational structure queries")

if __name__ == "__main__":
    test_heavy_light_decomposition()
    demonstrate_hld_applications()

"""
Heavy-Light Decomposition enables efficient path operations on trees
by decomposing into O(log n) heavy paths, supporting complex queries
in logarithmic time with segment tree integration.
"""
