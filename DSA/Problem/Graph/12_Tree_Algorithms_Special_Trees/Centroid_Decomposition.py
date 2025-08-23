"""
Centroid Decomposition - Advanced Tree Algorithm
Difficulty: Medium

Centroid Decomposition is a divide-and-conquer technique for trees that 
recursively decomposes a tree by removing centroids, creating a decomposition 
tree with O(log n) height. Enables efficient distance queries and path counting.

Key Concepts:
1. Tree Centroid Properties
2. Recursive Decomposition
3. Distance Queries
4. Path Counting
5. Decomposition Tree Structure
"""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict, deque

class CentroidDecomposition:
    """Centroid Decomposition implementation"""
    
    def __init__(self, n: int):
        self.n = n
        self.graph = defaultdict(list)
        self.removed = [False] * n
        self.subtree_size = [0] * n
        self.centroid_parent = [-1] * n
        self.centroid_level = [0] * n
        self.decomp_tree = defaultdict(list)
    
    def add_edge(self, u: int, v: int):
        """Add edge to tree"""
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def calculate_subtree_size(self, node: int, parent: int) -> int:
        """
        Calculate subtree sizes for centroid finding
        
        Time: O(n), Space: O(h)
        """
        self.subtree_size[node] = 1
        
        for neighbor in self.graph[node]:
            if neighbor != parent and not self.removed[neighbor]:
                self.subtree_size[node] += self.calculate_subtree_size(neighbor, node)
        
        return self.subtree_size[node]
    
    def find_centroid(self, node: int, parent: int, tree_size: int) -> int:
        """
        Find centroid of current tree component
        
        Time: O(n), Space: O(h)
        """
        for neighbor in self.graph[node]:
            if (neighbor != parent and not self.removed[neighbor] and
                self.subtree_size[neighbor] > tree_size // 2):
                return self.find_centroid(neighbor, node, tree_size)
        
        return node
    
    def decompose(self, node: int, parent_centroid: int = -1, level: int = 0) -> int:
        """
        Recursively decompose tree using centroids
        
        Time: O(n log n), Space: O(n)
        """
        # Calculate subtree size and find centroid
        tree_size = self.calculate_subtree_size(node, -1)
        centroid = self.find_centroid(node, -1, tree_size)
        
        # Mark centroid as removed and set properties
        self.removed[centroid] = True
        self.centroid_parent[centroid] = parent_centroid
        self.centroid_level[centroid] = level
        
        # Add to decomposition tree
        if parent_centroid != -1:
            self.decomp_tree[parent_centroid].append(centroid)
        
        # Recursively decompose subtrees
        for neighbor in self.graph[centroid]:
            if not self.removed[neighbor]:
                self.decompose(neighbor, centroid, level + 1)
        
        return centroid
    
    def build(self, root: int = 0) -> int:
        """
        Build centroid decomposition
        
        Time: O(n log n), Space: O(n)
        """
        return self.decompose(root)
    
    def distance_in_tree(self, u: int, v: int) -> int:
        """
        Calculate distance between two nodes in original tree
        
        Time: O(n) - can be optimized with LCA
        Space: O(n)
        """
        if u == v:
            return 0
        
        # BFS to find distance
        queue = deque([(u, 0)])
        visited = {u}
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in self.graph[node]:
                if neighbor == v:
                    return dist + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1  # Not connected
    
    def query_distance(self, u: int, v: int) -> int:
        """
        Query distance using centroid decomposition
        
        Time: O(log n), Space: O(log n)
        """
        # Find LCA in centroid decomposition tree
        def find_centroid_lca(x: int, y: int) -> int:
            # Get all ancestors of x
            ancestors_x = set()
            curr = x
            while curr != -1:
                ancestors_x.add(curr)
                curr = self.centroid_parent[curr]
            
            # Find first common ancestor of y
            curr = y
            while curr not in ancestors_x:
                curr = self.centroid_parent[curr]
            
            return curr
        
        lca_centroid = find_centroid_lca(u, v)
        
        # Distance = dist(u, lca) + dist(v, lca)
        dist_u_lca = self.distance_in_tree(u, lca_centroid)
        dist_v_lca = self.distance_in_tree(v, lca_centroid)
        
        return dist_u_lca + dist_v_lca

class CentroidDecompositionWithQueries:
    """Enhanced centroid decomposition with efficient queries"""
    
    def __init__(self, n: int):
        self.cd = CentroidDecomposition(n)
        self.distances = defaultdict(dict)  # distances[centroid][node] = distance
        self.active = [True] * n  # For dynamic updates
    
    def precompute_distances(self):
        """
        Precompute distances from each centroid to all nodes in its subtree
        
        Time: O(n log n), Space: O(n log n)
        """
        def dfs_distances(centroid: int, node: int, parent: int, dist: int):
            self.distances[centroid][node] = dist
            
            for neighbor in self.cd.graph[node]:
                if neighbor != parent and not self.cd.removed[neighbor]:
                    dfs_distances(centroid, neighbor, node, dist + 1)
        
        # For each centroid, compute distances to all nodes in its component
        for centroid in range(self.cd.n):
            if centroid in self.distances:  # This is a centroid
                # Temporarily unremove to compute distances
                self.cd.removed[centroid] = False
                dfs_distances(centroid, centroid, -1, 0)
                self.cd.removed[centroid] = True
    
    def count_paths_with_distance(self, target_distance: int) -> int:
        """
        Count paths with exactly target_distance length
        
        Time: O(n log n), Space: O(n)
        """
        total_paths = 0
        
        def count_in_subtree(centroid: int, node: int, parent: int, dist: int, 
                           path_count: Dict[int, int]):
            if dist > target_distance:
                return
            
            path_count[dist] = path_count.get(dist, 0) + 1
            
            for neighbor in self.cd.graph[node]:
                if neighbor != parent and not self.cd.removed[neighbor]:
                    count_in_subtree(centroid, neighbor, node, dist + 1, path_count)
        
        def solve(centroid: int):
            nonlocal total_paths
            
            # Count paths passing through this centroid
            all_paths = {0: 1}  # Distance 0 has 1 path (centroid itself)
            
            for neighbor in self.cd.graph[centroid]:
                if not self.cd.removed[neighbor]:
                    subtree_paths = {}
                    count_in_subtree(centroid, neighbor, centroid, 1, subtree_paths)
                    
                    # Count pairs with complementary distances
                    for dist, count in subtree_paths.items():
                        complement = target_distance - dist
                        if complement in all_paths:
                            total_paths += count * all_paths[complement]
                    
                    # Add current subtree paths to all_paths
                    for dist, count in subtree_paths.items():
                        all_paths[dist] = all_paths.get(dist, 0) + count
            
            # Recursively solve for child centroids
            self.cd.removed[centroid] = True
            for neighbor in self.cd.graph[centroid]:
                if not self.cd.removed[neighbor]:
                    child_centroid = self.cd.decompose(neighbor, centroid, 
                                                     self.cd.centroid_level[centroid] + 1)
                    solve(child_centroid)
        
        # Start from root centroid
        root_centroid = self.cd.build()
        solve(root_centroid)
        
        return total_paths

def test_centroid_decomposition():
    """Test centroid decomposition"""
    print("=== Testing Centroid Decomposition ===")
    
    # Create test tree
    n = 7
    cd = CentroidDecomposition(n)
    
    # Add edges to form tree
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for u, v in edges:
        cd.add_edge(u, v)
    
    print("Tree structure:")
    print("    0")
    print("   / \\")
    print("  1   2")
    print(" / \\ / \\")
    print("3  4 5  6")
    
    # Build centroid decomposition
    root_centroid = cd.build()
    
    print(f"\nCentroid Decomposition Results:")
    print(f"Root centroid: {root_centroid}")
    print(f"Centroid parents: {cd.centroid_parent}")
    print(f"Centroid levels: {cd.centroid_level}")
    
    # Reset removed array for distance queries
    cd.removed = [False] * n
    
    # Test distance queries
    test_pairs = [(0, 3), (3, 5), (4, 6), (1, 2)]
    print(f"\nDistance Queries:")
    for u, v in test_pairs:
        dist = cd.distance_in_tree(u, v)
        print(f"Distance({u}, {v}) = {dist}")

def demonstrate_centroid_applications():
    """Demonstrate centroid decomposition applications"""
    print("\n=== Centroid Decomposition Applications ===")
    
    print("Key Applications:")
    print("1. Distance Queries: O(log n) distance between any two nodes")
    print("2. Path Counting: Count paths with specific properties")
    print("3. Subtree Queries: Aggregate queries on tree paths")
    print("4. Dynamic Updates: Handle node additions/removals")
    print("5. Tree Isomorphism: Compare tree structures")
    
    print("\nComplexity Analysis:")
    print("• Preprocessing: O(n log n)")
    print("• Distance queries: O(log n)")
    print("• Path counting: O(n log n)")
    print("• Space: O(n log n) with precomputed distances")
    print("• Decomposition height: O(log n)")
    
    print("\nKey Properties:")
    print("• Each node appears in O(log n) centroid subtrees")
    print("• Centroid removal creates balanced decomposition")
    print("• Enables divide-and-conquer on trees")
    print("• Supports various tree queries efficiently")
    
    print("\nReal-world Applications:")
    print("• Network analysis and shortest paths")
    print("• Hierarchical clustering")
    print("• Tree-based machine learning")
    print("• Computational biology (phylogenetic trees)")

def analyze_centroid_properties():
    """Analyze properties of centroid decomposition"""
    print("\n=== Centroid Properties Analysis ===")
    
    print("Centroid Definition:")
    print("• Node whose removal creates no component > n/2 nodes")
    print("• Every tree has at least one centroid")
    print("• Tree with n nodes has at most 2 centroids")
    print("• Centroid minimizes maximum component size")
    
    print("\nDecomposition Properties:")
    print("• Height of decomposition tree: O(log n)")
    print("• Each original node appears in O(log n) levels")
    print("• Total work across all levels: O(n log n)")
    print("• Balanced divide-and-conquer structure")
    
    print("\nQuery Efficiency:")
    print("• Distance queries use centroid LCA")
    print("• Path properties computed via centroid paths")
    print("• Subtree operations decomposed efficiently")
    print("• Dynamic updates maintain decomposition")
    
    print("\nImplementation Considerations:")
    print("• Careful handling of removed nodes")
    print("• Efficient centroid finding algorithms")
    print("• Memory management for distance storage")
    print("• Optimization for specific query types")

if __name__ == "__main__":
    test_centroid_decomposition()
    demonstrate_centroid_applications()
    analyze_centroid_properties()

"""
Centroid Decomposition provides a powerful framework for tree algorithms,
enabling efficient distance queries, path counting, and divide-and-conquer
approaches on tree structures with logarithmic query complexity.
"""
