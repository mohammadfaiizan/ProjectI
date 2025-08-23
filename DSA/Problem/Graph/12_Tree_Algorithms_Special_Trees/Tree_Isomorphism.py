"""
Tree Isomorphism - Advanced Algorithm
Difficulty: Hard

Tree isomorphism determines if two trees have the same structure.
Two trees are isomorphic if there exists a bijection between their vertices
that preserves adjacency relationships.

Key Concepts:
1. Canonical Tree Representation
2. Tree Hashing and Fingerprinting
3. AHU (Aho, Hopcroft, Ullman) Algorithm
4. Rooted vs Unrooted Tree Isomorphism
5. Tree Automorphism Groups
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import hashlib

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.children = []

class TreeIsomorphism:
    """Tree isomorphism algorithms and utilities"""
    
    def __init__(self):
        self.hash_memo = {}
    
    def are_isomorphic_ahu(self, tree1: List[List[int]], tree2: List[List[int]]) -> bool:
        """
        Approach 1: AHU Algorithm for Tree Isomorphism
        
        Uses canonical labeling to determine isomorphism.
        
        Time: O(n log n), Space: O(n)
        """
        if len(tree1) != len(tree2):
            return False
        
        n = len(tree1)
        if n == 0:
            return True
        
        # Build adjacency lists
        adj1 = self._build_adjacency_list(tree1)
        adj2 = self._build_adjacency_list(tree2)
        
        # Get canonical forms
        canon1 = self._get_canonical_form(adj1, n)
        canon2 = self._get_canonical_form(adj2, n)
        
        return canon1 == canon2
    
    def are_isomorphic_hash(self, tree1: List[List[int]], tree2: List[List[int]]) -> bool:
        """
        Approach 2: Tree Hashing Method
        
        Uses recursive hashing to compare tree structures.
        
        Time: O(n), Space: O(n)
        """
        if len(tree1) != len(tree2):
            return False
        
        n = len(tree1)
        if n == 0:
            return True
        
        adj1 = self._build_adjacency_list(tree1)
        adj2 = self._build_adjacency_list(tree2)
        
        # Try all possible roots for both trees
        hashes1 = set()
        hashes2 = set()
        
        for root in range(n):
            if root in adj1:
                hash1 = self._compute_tree_hash(adj1, root, -1)
                hashes1.add(hash1)
        
        for root in range(n):
            if root in adj2:
                hash2 = self._compute_tree_hash(adj2, root, -1)
                hashes2.add(hash2)
        
        return len(hashes1.intersection(hashes2)) > 0
    
    def are_isomorphic_center_based(self, tree1: List[List[int]], tree2: List[List[int]]) -> bool:
        """
        Approach 3: Center-Based Isomorphism Check
        
        Uses tree centers to reduce the number of root candidates.
        
        Time: O(n), Space: O(n)
        """
        if len(tree1) != len(tree2):
            return False
        
        n = len(tree1)
        if n == 0:
            return True
        
        adj1 = self._build_adjacency_list(tree1)
        adj2 = self._build_adjacency_list(tree2)
        
        # Find centers of both trees
        centers1 = self._find_tree_centers(adj1, n)
        centers2 = self._find_tree_centers(adj2, n)
        
        # Compare canonical forms rooted at centers
        for c1 in centers1:
            hash1 = self._compute_tree_hash(adj1, c1, -1)
            for c2 in centers2:
                hash2 = self._compute_tree_hash(adj2, c2, -1)
                if hash1 == hash2:
                    return True
        
        return False
    
    def are_isomorphic_degree_sequence(self, tree1: List[List[int]], tree2: List[List[int]]) -> bool:
        """
        Approach 4: Degree Sequence + Structural Check
        
        First checks degree sequences, then structural equivalence.
        
        Time: O(n log n), Space: O(n)
        """
        if len(tree1) != len(tree2):
            return False
        
        n = len(tree1)
        if n == 0:
            return True
        
        adj1 = self._build_adjacency_list(tree1)
        adj2 = self._build_adjacency_list(tree2)
        
        # Check degree sequences
        degrees1 = sorted([len(adj1[i]) for i in range(n)])
        degrees2 = sorted([len(adj2[i]) for i in range(n)])
        
        if degrees1 != degrees2:
            return False
        
        # If degree sequences match, check structural isomorphism
        return self.are_isomorphic_center_based(tree1, tree2)
    
    def find_isomorphism_mapping(self, tree1: List[List[int]], tree2: List[List[int]]) -> Optional[Dict[int, int]]:
        """
        Approach 5: Find Explicit Isomorphism Mapping
        
        Returns the actual vertex mapping if trees are isomorphic.
        
        Time: O(n!), Space: O(n) - worst case, often much better
        """
        if len(tree1) != len(tree2):
            return None
        
        n = len(tree1)
        if n == 0:
            return {}
        
        adj1 = self._build_adjacency_list(tree1)
        adj2 = self._build_adjacency_list(tree2)
        
        # Find centers to reduce search space
        centers1 = self._find_tree_centers(adj1, n)
        centers2 = self._find_tree_centers(adj2, n)
        
        for c1 in centers1:
            for c2 in centers2:
                mapping = self._find_mapping_rooted(adj1, adj2, c1, c2, n)
                if mapping:
                    return mapping
        
        return None
    
    def count_automorphisms(self, tree: List[List[int]]) -> int:
        """
        Approach 6: Count Tree Automorphisms
        
        Counts the number of isomorphisms from tree to itself.
        
        Time: O(n^2), Space: O(n)
        """
        n = len(tree)
        if n == 0:
            return 1
        
        adj = self._build_adjacency_list(tree)
        centers = self._find_tree_centers(adj, n)
        
        total_automorphisms = 0
        
        for center in centers:
            # Count automorphisms rooted at this center
            automorphisms = self._count_rooted_automorphisms(adj, center, -1)
            total_automorphisms += automorphisms
        
        # If tree has 2 centers, we've double-counted
        if len(centers) == 2:
            total_automorphisms //= 2
        
        return total_automorphisms
    
    def _build_adjacency_list(self, edges: List[List[int]]) -> Dict[int, List[int]]:
        """Build adjacency list from edge list"""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def _get_canonical_form(self, adj: Dict[int, List[int]], n: int) -> str:
        """Get canonical form of tree using AHU algorithm"""
        # Find tree center(s)
        centers = self._find_tree_centers(adj, n)
        
        # Get canonical form rooted at center
        canonical_forms = []
        for center in centers:
            form = self._get_canonical_subtree(adj, center, -1)
            canonical_forms.append(form)
        
        return min(canonical_forms)  # Lexicographically smallest
    
    def _get_canonical_subtree(self, adj: Dict[int, List[int]], node: int, parent: int) -> str:
        """Get canonical representation of subtree rooted at node"""
        children_forms = []
        
        for child in adj[node]:
            if child != parent:
                child_form = self._get_canonical_subtree(adj, child, node)
                children_forms.append(child_form)
        
        children_forms.sort()  # Sort for canonical ordering
        return "(" + "".join(children_forms) + ")"
    
    def _compute_tree_hash(self, adj: Dict[int, List[int]], node: int, parent: int) -> int:
        """Compute hash of subtree rooted at node"""
        if (node, parent) in self.hash_memo:
            return self.hash_memo[(node, parent)]
        
        child_hashes = []
        for child in adj[node]:
            if child != parent:
                child_hash = self._compute_tree_hash(adj, child, node)
                child_hashes.append(child_hash)
        
        child_hashes.sort()  # Sort for canonical ordering
        
        # Combine hashes
        hash_str = str(child_hashes)
        hash_val = hash(hash_str) % (10**9 + 7)
        
        self.hash_memo[(node, parent)] = hash_val
        return hash_val
    
    def _find_tree_centers(self, adj: Dict[int, List[int]], n: int) -> List[int]:
        """Find center(s) of tree"""
        if n == 1:
            return [0]
        
        # Initialize degrees
        degrees = {i: len(adj[i]) for i in range(n)}
        leaves = deque([i for i in range(n) if degrees[i] <= 1])
        
        remaining = n
        
        while remaining > 2:
            leaf_count = len(leaves)
            remaining -= leaf_count
            
            for _ in range(leaf_count):
                leaf = leaves.popleft()
                
                for neighbor in adj[leaf]:
                    degrees[neighbor] -= 1
                    if degrees[neighbor] == 1:
                        leaves.append(neighbor)
        
        return [i for i in range(n) if degrees[i] > 0]
    
    def _find_mapping_rooted(self, adj1: Dict[int, List[int]], adj2: Dict[int, List[int]], 
                           root1: int, root2: int, n: int) -> Optional[Dict[int, int]]:
        """Find isomorphism mapping between rooted trees"""
        mapping = {}
        
        def dfs_match(node1: int, node2: int, parent1: int, parent2: int) -> bool:
            mapping[node1] = node2
            
            children1 = [child for child in adj1[node1] if child != parent1]
            children2 = [child for child in adj2[node2] if child != parent2]
            
            if len(children1) != len(children2):
                return False
            
            # Sort children by their subtree canonical forms
            children1.sort(key=lambda x: self._compute_tree_hash(adj1, x, node1))
            children2.sort(key=lambda x: self._compute_tree_hash(adj2, x, node2))
            
            for c1, c2 in zip(children1, children2):
                if not dfs_match(c1, c2, node1, node2):
                    return False
            
            return True
        
        if dfs_match(root1, root2, -1, -1):
            return mapping
        
        return None
    
    def _count_rooted_automorphisms(self, adj: Dict[int, List[int]], node: int, parent: int) -> int:
        """Count automorphisms of subtree rooted at node"""
        children = [child for child in adj[node] if child != parent]
        
        if not children:
            return 1
        
        # Group children by their canonical forms
        child_groups = defaultdict(list)
        for child in children:
            canonical_form = self._get_canonical_subtree(adj, child, node)
            child_groups[canonical_form].append(child)
        
        total_automorphisms = 1
        
        for group in child_groups.values():
            group_size = len(group)
            
            # Automorphisms within the group
            group_automorphisms = 1
            for child in group:
                child_automorphisms = self._count_rooted_automorphisms(adj, child, node)
                group_automorphisms *= child_automorphisms
            
            # Permutations of identical subtrees
            import math
            group_automorphisms *= math.factorial(group_size)
            
            total_automorphisms *= group_automorphisms
        
        return total_automorphisms

def test_tree_isomorphism():
    """Test tree isomorphism algorithms"""
    solver = TreeIsomorphism()
    
    # Test cases: (tree1, tree2, expected, description)
    test_cases = [
        # Isomorphic trees
        ([[0,1],[1,2],[2,3]], [[0,1],[1,2],[1,3]], True, "Path vs Star"),
        ([[0,1],[0,2],[1,3],[1,4]], [[0,1],[0,2],[2,3],[2,4]], True, "Isomorphic trees"),
        
        # Non-isomorphic trees
        ([[0,1],[1,2],[2,3]], [[0,1],[0,2],[0,3]], False, "Path vs Star (different)"),
        ([[0,1],[1,2]], [[0,1],[1,2],[2,3]], False, "Different sizes"),
        
        # Edge cases
        ([], [], True, "Empty trees"),
        ([[0,1]], [[0,1]], True, "Single edge"),
    ]
    
    algorithms = [
        ("AHU Algorithm", solver.are_isomorphic_ahu),
        ("Tree Hashing", solver.are_isomorphic_hash),
        ("Center-Based", solver.are_isomorphic_center_based),
        ("Degree Sequence", solver.are_isomorphic_degree_sequence),
    ]
    
    print("=== Testing Tree Isomorphism ===")
    
    for tree1, tree2, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(tree1, tree2)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:15} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:15} | ERROR: {str(e)[:30]}")
    
    # Test isomorphism mapping
    print(f"\n--- Isomorphism Mapping Demo ---")
    tree1 = [[0,1],[0,2],[1,3],[1,4]]
    tree2 = [[0,1],[0,2],[2,3],[2,4]]
    
    mapping = solver.find_isomorphism_mapping(tree1, tree2)
    if mapping:
        print(f"Mapping found: {mapping}")
    else:
        print("No mapping found")
    
    # Test automorphism counting
    print(f"\n--- Automorphism Counting Demo ---")
    symmetric_tree = [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6]]
    automorphisms = solver.count_automorphisms(symmetric_tree)
    print(f"Automorphisms in symmetric tree: {automorphisms}")

def demonstrate_isomorphism_applications():
    """Demonstrate applications of tree isomorphism"""
    print("\n=== Tree Isomorphism Applications ===")
    
    print("Key Applications:")
    print("1. Chemical Structure Analysis: Molecular isomorphism")
    print("2. Phylogenetic Tree Comparison: Evolutionary relationships")
    print("3. Network Analysis: Structural equivalence")
    print("4. Compiler Optimization: Expression tree equivalence")
    print("5. Database Query Optimization: Query plan comparison")
    
    print("\nComplexity Analysis:")
    print("• AHU Algorithm: O(n log n) time, O(n) space")
    print("• Tree Hashing: O(n) time, O(n) space")
    print("• Center-Based: O(n) time, O(n) space")
    print("• Mapping Finding: O(n!) worst case, often much better")
    
    print("\nPractical Considerations:")
    print("• Tree hashing is fastest for most cases")
    print("• AHU algorithm provides canonical labeling")
    print("• Center-based approach reduces search space")
    print("• Explicit mapping useful for transformation")

if __name__ == "__main__":
    test_tree_isomorphism()
    demonstrate_isomorphism_applications()

"""
Tree Isomorphism algorithms provide sophisticated methods for comparing
tree structures, with applications in chemistry, biology, computer science,
and network analysis. Different approaches offer various trade-offs between
speed, accuracy, and additional information provided.
"""
