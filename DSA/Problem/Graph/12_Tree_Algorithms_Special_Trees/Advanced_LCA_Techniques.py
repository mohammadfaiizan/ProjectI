"""
Advanced LCA Techniques - Comprehensive Implementation
Difficulty: Hard

This file implements advanced techniques for Lowest Common Ancestor (LCA) queries
including Binary Lifting, Euler Tour + RMQ, and other sophisticated approaches
for efficient LCA computation in various scenarios.

Key Concepts:
1. Binary Lifting (Sparse Table)
2. Euler Tour + Range Minimum Query
3. Tarjan's Offline LCA
4. Heavy-Light Decomposition LCA
5. LCA with Weighted Paths
6. Dynamic LCA (Link-Cut Trees)
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import math

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.children = []

class BinaryLiftingLCA:
    """Binary Lifting approach for LCA queries"""
    
    def __init__(self, n: int):
        self.n = n
        self.LOG = int(math.log2(n)) + 1
        self.graph = defaultdict(list)
        self.parent = [[-1] * self.LOG for _ in range(n)]
        self.depth = [0] * n
        self.built = False
    
    def add_edge(self, u: int, v: int):
        """Add edge to tree"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def build(self, root: int = 0):
        """
        Build binary lifting table
        
        Time: O(n log n), Space: O(n log n)
        """
        # DFS to set depths and direct parents
        def dfs(node: int, par: int, d: int):
            self.parent[node][0] = par
            self.depth[node] = d
            
            for child in self.graph[node]:
                if child != par:
                    dfs(child, node, d + 1)
        
        dfs(root, -1, 0)
        
        # Fill binary lifting table
        for j in range(1, self.LOG):
            for i in range(self.n):
                if self.parent[i][j-1] != -1:
                    self.parent[i][j] = self.parent[self.parent[i][j-1]][j-1]
        
        self.built = True
    
    def lca(self, u: int, v: int) -> int:
        """
        Find LCA using binary lifting
        
        Time: O(log n), Space: O(1)
        """
        if not self.built:
            raise ValueError("Must call build() first")
        
        # Make u deeper than v
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        
        # Bring u to same level as v
        diff = self.depth[u] - self.depth[v]
        for i in range(self.LOG):
            if (diff >> i) & 1:
                u = self.parent[u][i]
        
        if u == v:
            return u
        
        # Binary search for LCA
        for i in range(self.LOG - 1, -1, -1):
            if self.parent[u][i] != self.parent[v][i]:
                u = self.parent[u][i]
                v = self.parent[v][i]
        
        return self.parent[u][0]
    
    def distance(self, u: int, v: int) -> int:
        """Calculate distance between two nodes"""
        lca_node = self.lca(u, v)
        return self.depth[u] + self.depth[v] - 2 * self.depth[lca_node]
    
    def kth_ancestor(self, node: int, k: int) -> int:
        """Find k-th ancestor of node"""
        if k > self.depth[node]:
            return -1
        
        for i in range(self.LOG):
            if (k >> i) & 1:
                node = self.parent[node][i]
                if node == -1:
                    return -1
        
        return node

class EulerTourLCA:
    """Euler Tour + RMQ approach for LCA"""
    
    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.euler_tour = []
        self.first_occurrence = {}
        self.depth = []
        self.sparse_table = []
        self.built = False
    
    def add_edge(self, u: int, v: int):
        """Add edge to tree"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def build(self, root: int = 0):
        """
        Build Euler tour and sparse table for RMQ
        
        Time: O(n log n), Space: O(n log n)
        """
        # Perform Euler tour
        def euler_dfs(node: int, parent: int, d: int):
            if node not in self.first_occurrence:
                self.first_occurrence[node] = len(self.euler_tour)
            
            self.euler_tour.append(node)
            self.depth.append(d)
            
            for child in self.graph[node]:
                if child != parent:
                    euler_dfs(child, node, d + 1)
                    self.euler_tour.append(node)
                    self.depth.append(d)
        
        euler_dfs(root, -1, 0)
        
        # Build sparse table for RMQ on depth array
        m = len(self.depth)
        LOG = int(math.log2(m)) + 1
        self.sparse_table = [[0] * LOG for _ in range(m)]
        
        # Initialize for length 1
        for i in range(m):
            self.sparse_table[i][0] = i
        
        # Fill sparse table
        j = 1
        while (1 << j) <= m:
            i = 0
            while (i + (1 << j) - 1) < m:
                left = self.sparse_table[i][j-1]
                right = self.sparse_table[i + (1 << (j-1))][j-1]
                
                if self.depth[left] < self.depth[right]:
                    self.sparse_table[i][j] = left
                else:
                    self.sparse_table[i][j] = right
                
                i += 1
            j += 1
        
        self.built = True
    
    def rmq(self, l: int, r: int) -> int:
        """Range minimum query on depth array"""
        length = r - l + 1
        k = int(math.log2(length))
        
        left = self.sparse_table[l][k]
        right = self.sparse_table[r - (1 << k) + 1][k]
        
        if self.depth[left] < self.depth[right]:
            return left
        else:
            return right
    
    def lca(self, u: int, v: int) -> int:
        """
        Find LCA using Euler tour + RMQ
        
        Time: O(1), Space: O(1)
        """
        if not self.built:
            raise ValueError("Must call build() first")
        
        left = self.first_occurrence[u]
        right = self.first_occurrence[v]
        
        if left > right:
            left, right = right, left
        
        min_depth_idx = self.rmq(left, right)
        return self.euler_tour[min_depth_idx]

class TarjanOfflineLCA:
    """Tarjan's offline LCA algorithm using Union-Find"""
    
    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.parent = list(range(n))
        self.rank = [0] * n
        self.ancestor = list(range(n))
        self.visited = [False] * n
        self.queries = defaultdict(list)
        self.answers = {}
    
    def add_edge(self, u: int, v: int):
        """Add edge to tree"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def add_query(self, u: int, v: int, query_id: int):
        """Add LCA query"""
        self.queries[u].append((v, query_id))
        self.queries[v].append((u, query_id))
    
    def find(self, x: int) -> int:
        """Union-Find find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """Union-Find union by rank"""
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def tarjan_lca(self, node: int, parent: int):
        """
        Tarjan's LCA algorithm
        
        Time: O(n + q * α(n)), Space: O(n)
        """
        self.ancestor[self.find(node)] = node
        
        for child in self.graph[node]:
            if child != parent:
                self.tarjan_lca(child, node)
                self.union(node, child)
                self.ancestor[self.find(node)] = node
        
        self.visited[node] = True
        
        # Process queries involving this node
        for other_node, query_id in self.queries[node]:
            if self.visited[other_node]:
                lca_node = self.ancestor[self.find(other_node)]
                self.answers[query_id] = lca_node
    
    def solve_queries(self, root: int = 0) -> Dict[int, int]:
        """
        Solve all LCA queries offline
        
        Time: O(n + q * α(n)), Space: O(n + q)
        """
        self.tarjan_lca(root, -1)
        return self.answers

class WeightedPathLCA:
    """LCA with weighted paths and distance queries"""
    
    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.binary_lifting = BinaryLiftingLCA(n)
        self.dist_from_root = [0] * n
        self.built = False
    
    def add_edge(self, u: int, v: int, weight: int = 1):
        """Add weighted edge to tree"""
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))
        self.binary_lifting.add_edge(u, v)
    
    def build(self, root: int = 0):
        """Build weighted LCA structure"""
        # Calculate distances from root
        def dfs(node: int, parent: int, dist: int):
            self.dist_from_root[node] = dist
            
            for neighbor, weight in self.graph[node]:
                if neighbor != parent:
                    dfs(neighbor, node, dist + weight)
        
        dfs(root, -1, 0)
        self.binary_lifting.build(root)
        self.built = True
    
    def lca(self, u: int, v: int) -> int:
        """Find LCA"""
        return self.binary_lifting.lca(u, v)
    
    def weighted_distance(self, u: int, v: int) -> int:
        """Calculate weighted distance between two nodes"""
        if not self.built:
            raise ValueError("Must call build() first")
        
        lca_node = self.lca(u, v)
        return (self.dist_from_root[u] + self.dist_from_root[v] - 
                2 * self.dist_from_root[lca_node])

def test_advanced_lca():
    """Test advanced LCA techniques"""
    print("=== Testing Advanced LCA Techniques ===")
    
    # Create test tree
    n = 8
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7)]
    
    print("Tree structure:")
    print("      0")
    print("     / \\")
    print("    1   2")
    print("   /|   |\\")
    print("  3 4   5 6")
    print("  |")
    print("  7")
    
    # Test Binary Lifting
    print(f"\n--- Binary Lifting LCA ---")
    bl_lca = BinaryLiftingLCA(n)
    for u, v in edges:
        bl_lca.add_edge(u, v)
    bl_lca.build(0)
    
    test_pairs = [(3, 4), (7, 5), (4, 6), (1, 2)]
    for u, v in test_pairs:
        lca_result = bl_lca.lca(u, v)
        distance = bl_lca.distance(u, v)
        print(f"LCA({u}, {v}) = {lca_result}, Distance = {distance}")
    
    # Test Euler Tour LCA
    print(f"\n--- Euler Tour + RMQ LCA ---")
    et_lca = EulerTourLCA(n)
    for u, v in edges:
        et_lca.add_edge(u, v)
    et_lca.build(0)
    
    for u, v in test_pairs:
        lca_result = et_lca.lca(u, v)
        print(f"LCA({u}, {v}) = {lca_result}")
    
    # Test Tarjan Offline LCA
    print(f"\n--- Tarjan Offline LCA ---")
    tarjan_lca = TarjanOfflineLCA(n)
    for u, v in edges:
        tarjan_lca.add_edge(u, v)
    
    # Add queries
    for i, (u, v) in enumerate(test_pairs):
        tarjan_lca.add_query(u, v, i)
    
    answers = tarjan_lca.solve_queries(0)
    for i, (u, v) in enumerate(test_pairs):
        print(f"LCA({u}, {v}) = {answers[i]}")

def analyze_lca_complexity():
    """Analyze complexity of different LCA approaches"""
    print("\n=== LCA Complexity Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Binary Lifting:**")
    print("   • Preprocessing: O(n log n)")
    print("   • Query: O(log n)")
    print("   • Space: O(n log n)")
    print("   • Pros: Simple, supports k-th ancestor")
    print("   • Cons: Higher space complexity")
    
    print("\n2. **Euler Tour + RMQ:**")
    print("   • Preprocessing: O(n log n)")
    print("   • Query: O(1)")
    print("   • Space: O(n log n)")
    print("   • Pros: Optimal query time")
    print("   • Cons: Complex implementation")
    
    print("\n3. **Tarjan Offline:**")
    print("   • Preprocessing: O(n + q * α(n))")
    print("   • Query: O(1) per query (offline)")
    print("   • Space: O(n + q)")
    print("   • Pros: Optimal for batch queries")
    print("   • Cons: Offline only")
    
    print("\n4. **Heavy-Light Decomposition:**")
    print("   • Preprocessing: O(n)")
    print("   • Query: O(log n)")
    print("   • Space: O(n)")
    print("   • Pros: Supports path queries")
    print("   • Cons: More complex for simple LCA")
    
    print("\nRecommendations:")
    print("• Online queries with O(1): Euler Tour + RMQ")
    print("• Simple implementation: Binary Lifting")
    print("• Batch queries: Tarjan Offline")
    print("• Path operations: Heavy-Light Decomposition")

if __name__ == "__main__":
    test_advanced_lca()
    analyze_lca_complexity()

"""
Advanced LCA Techniques provide sophisticated approaches for ancestor queries
in trees, each optimized for different scenarios and query patterns with
varying preprocessing and query time complexities.
"""
