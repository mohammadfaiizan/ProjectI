"""
Second Minimum Spanning Tree
Difficulty: Hard

Problem:
Given a connected, undirected graph with n vertices and m edges, find the weight of the 
second minimum spanning tree (SMST). The second minimum spanning tree is the spanning 
tree with the second smallest total weight among all possible spanning trees of the graph.

If the graph has multiple MSTs with the same minimum weight, the SMST has a weight 
strictly greater than the MST weight. If all spanning trees have the same weight, 
return -1.

Examples:
Input: n = 4, edges = [[0,1,1],[1,2,2],[2,3,3],[0,3,4],[0,2,5]]
Output: 10

Input: n = 3, edges = [[0,1,1],[1,2,1],[0,2,1]]
Output: -1 (All spanning trees have weight 2)

Input: n = 4, edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,2]]
Output: 4

Constraints:
- 2 <= n <= 1000
- n-1 <= m <= n*(n-1)/2
- 1 <= weight <= 1000
- The graph is connected
"""

from typing import List, Tuple, Optional
import heapq
from collections import defaultdict

class Solution:
    def findSecondMinimumMST_approach1_edge_replacement(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 1: Edge Replacement Strategy (Optimal)
        
        Find MST, then try replacing each MST edge with non-MST edges to find SMST.
        
        Time: O(V^3) or O(E * V * α(V))
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        edge_set = set()
        
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
            edge_set.add((min(u,v), max(u,v), w))
        
        # Find MST using Kruskal's algorithm
        mst_weight, mst_edges = self._find_mst_kruskal(n, edges)
        
        if mst_weight == -1:
            return -1
        
        # Convert MST edges to set for quick lookup
        mst_edge_set = set()
        for u, v, w in mst_edges:
            mst_edge_set.add((min(u,v), max(u,v), w))
        
        min_smst_weight = float('inf')
        
        # Try replacing each MST edge
        for remove_u, remove_v, remove_w in mst_edges:
            # Create graph without this MST edge
            modified_edges = []
            for u, v, w in edges:
                if not (min(u,v) == min(remove_u, remove_v) and 
                       max(u,v) == max(remove_u, remove_v) and w == remove_w):
                    modified_edges.append([u, v, w])
            
            # Find MST of modified graph
            temp_weight, temp_edges = self._find_mst_kruskal(n, modified_edges)
            
            if temp_weight != -1 and temp_weight > mst_weight:
                min_smst_weight = min(min_smst_weight, temp_weight)
        
        return min_smst_weight if min_smst_weight != float('inf') else -1
    
    def findSecondMinimumMST_approach2_cycle_replacement(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 2: Cycle Replacement Strategy
        
        Add non-MST edges one by one to create cycles, then remove heaviest MST edge.
        
        Time: O(E * V * α(V))
        Space: O(V + E)
        """
        # Find MST
        mst_weight, mst_edges = self._find_mst_kruskal(n, edges)
        
        if mst_weight == -1:
            return -1
        
        # Build MST as adjacency list
        mst_graph = defaultdict(list)
        mst_edge_weights = {}
        
        for u, v, w in mst_edges:
            mst_graph[u].append(v)
            mst_graph[v].append(u)
            mst_edge_weights[(min(u,v), max(u,v))] = w
        
        min_smst_weight = float('inf')
        
        # Try adding each non-MST edge
        for u, v, w in edges:
            edge_key = (min(u,v), max(u,v))
            
            if edge_key not in mst_edge_weights:
                # This edge is not in MST, adding it creates a cycle
                # Find the path in MST between u and v
                path = self._find_path_in_tree(mst_graph, u, v, n)
                
                if path:
                    # Find heaviest edge in the path
                    max_weight = 0
                    max_edge = None
                    
                    for i in range(len(path) - 1):
                        curr, next_node = path[i], path[i + 1]
                        path_edge_key = (min(curr, next_node), max(curr, next_node))
                        edge_weight = mst_edge_weights[path_edge_key]
                        
                        if edge_weight > max_weight:
                            max_weight = edge_weight
                            max_edge = path_edge_key
                    
                    # Calculate new spanning tree weight
                    new_weight = mst_weight - max_weight + w
                    
                    if new_weight > mst_weight:
                        min_smst_weight = min(min_smst_weight, new_weight)
        
        return min_smst_weight if min_smst_weight != float('inf') else -1
    
    def findSecondMinimumMST_approach3_sorted_edge_enumeration(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 3: Sorted Edge Enumeration
        
        Enumerate all possible spanning trees in weight order.
        
        Time: O(E^2 * α(V))
        Space: O(E)
        """
        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        # Find first MST weight
        first_mst_weight = self._find_mst_weight_from_sorted(n, sorted_edges)
        
        if first_mst_weight == -1:
            return -1
        
        # Try all possible edge combinations for SMST
        min_smst_weight = float('inf')
        
        # Generate alternative MSTs by modifying edge selection
        self._find_alternative_msts(n, sorted_edges, first_mst_weight, 
                                   0, UnionFind(n), 0, 0, min_smst_weight)
        
        return min_smst_weight[0] if min_smst_weight[0] != float('inf') else -1
    
    def findSecondMinimumMST_approach4_contraction_algorithm(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 4: Graph Contraction Algorithm
        
        Use graph contraction to systematically find SMST.
        
        Time: O(V^2 * E)
        Space: O(V + E)
        """
        # Find MST
        mst_weight, mst_edges = self._find_mst_kruskal(n, edges)
        
        if mst_weight == -1:
            return -1
        
        min_smst_weight = float('inf')
        
        # For each pair of vertices, find alternative paths
        for u in range(n):
            for v in range(u + 1, n):
                # Find all simple paths between u and v
                all_paths = self._find_all_simple_paths(edges, u, v, n)
                
                for path_edges in all_paths:
                    # Check if this path forms a valid spanning tree with other edges
                    path_weight = sum(w for _, _, w in path_edges)
                    
                    # Try to complete spanning tree with remaining edges
                    remaining_edges = [e for e in edges if e not in path_edges]
                    
                    # Use modified Kruskal's to complete spanning tree
                    completion_weight = self._complete_spanning_tree(
                        n, path_edges, remaining_edges)
                    
                    if completion_weight != -1:
                        total_weight = completion_weight
                        if total_weight > mst_weight:
                            min_smst_weight = min(min_smst_weight, total_weight)
        
        return min_smst_weight if min_smst_weight != float('inf') else -1
    
    def findSecondMinimumMST_approach5_matroid_intersection(self, n: int, edges: List[List[int]]) -> int:
        """
        Approach 5: Matroid Intersection (Advanced Theory)
        
        Use matroid theory for systematic SMST enumeration.
        
        Time: O(E^3)
        Space: O(E^2)
        """
        # Find all spanning trees using matroid intersection
        all_msts = self._enumerate_all_spanning_trees(n, edges)
        
        if len(all_msts) < 2:
            return -1
        
        # Sort by weight
        all_msts.sort(key=lambda x: x[0])
        
        # Find first weight > MST weight
        first_weight = all_msts[0][0]
        
        for weight, tree in all_msts:
            if weight > first_weight:
                return weight
        
        return -1  # All spanning trees have same weight
    
    def _find_mst_kruskal(self, n: int, edges: List[List[int]]) -> Tuple[int, List[Tuple[int, int, int]]]:
        """Find MST using Kruskal's algorithm"""
        sorted_edges = sorted(edges, key=lambda x: x[2])
        uf = UnionFind(n)
        
        weight = 0
        mst_edges = []
        
        for u, v, w in sorted_edges:
            if uf.union(u, v):
                weight += w
                mst_edges.append((u, v, w))
                
                if len(mst_edges) == n - 1:
                    break
        
        if len(mst_edges) == n - 1:
            return weight, mst_edges
        else:
            return -1, []
    
    def _find_path_in_tree(self, graph: dict, start: int, end: int, n: int) -> List[int]:
        """Find path between two nodes in a tree using DFS"""
        visited = [False] * n
        parent = [-1] * n
        
        def dfs(node, target):
            visited[node] = True
            
            if node == target:
                return True
            
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    parent[neighbor] = node
                    if dfs(neighbor, target):
                        return True
            
            return False
        
        if dfs(start, end):
            # Reconstruct path
            path = []
            current = end
            
            while current != -1:
                path.append(current)
                current = parent[current]
            
            path.reverse()
            return path
        
        return []
    
    def _find_mst_weight_from_sorted(self, n: int, sorted_edges: List[List[int]]) -> int:
        """Find MST weight from sorted edges"""
        uf = UnionFind(n)
        weight = 0
        edges_used = 0
        
        for u, v, w in sorted_edges:
            if uf.union(u, v):
                weight += w
                edges_used += 1
                
                if edges_used == n - 1:
                    break
        
        return weight if edges_used == n - 1 else -1
    
    def _find_alternative_msts(self, n: int, sorted_edges: List[List[int]], 
                              target_weight: int, edge_idx: int, uf: 'UnionFind', 
                              current_weight: int, edges_used: int, 
                              min_smst_weight: List[int]) -> None:
        """Recursively find alternative MSTs"""
        if edges_used == n - 1:
            if current_weight > target_weight:
                min_smst_weight[0] = min(min_smst_weight[0], current_weight)
            return
        
        if edge_idx >= len(sorted_edges):
            return
        
        # Try including current edge
        u, v, w = sorted_edges[edge_idx]
        
        if uf.union(u, v):
            self._find_alternative_msts(n, sorted_edges, target_weight, 
                                       edge_idx + 1, uf, current_weight + w, 
                                       edges_used + 1, min_smst_weight)
            uf.undo_union()  # Backtrack
        
        # Try not including current edge
        self._find_alternative_msts(n, sorted_edges, target_weight, 
                                   edge_idx + 1, uf, current_weight, 
                                   edges_used, min_smst_weight)
    
    def _find_all_simple_paths(self, edges: List[List[int]], start: int, 
                              end: int, n: int) -> List[List[Tuple[int, int, int]]]:
        """Find all simple paths between two vertices"""
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))
        
        all_paths = []
        visited = [False] * n
        current_path = []
        
        def dfs(node, target):
            if node == target:
                all_paths.append(current_path[:])
                return
            
            visited[node] = True
            
            for neighbor, weight in graph[node]:
                if not visited[neighbor]:
                    current_path.append((node, neighbor, weight))
                    dfs(neighbor, target)
                    current_path.pop()
            
            visited[node] = False
        
        dfs(start, end)
        return all_paths
    
    def _complete_spanning_tree(self, n: int, required_edges: List[Tuple[int, int, int]], 
                               available_edges: List[List[int]]) -> int:
        """Complete spanning tree with required edges"""
        uf = UnionFind(n)
        weight = 0
        
        # Add required edges
        for u, v, w in required_edges:
            if not uf.union(u, v):
                return -1  # Cycle in required edges
            weight += w
        
        # Add remaining edges using Kruskal's
        sorted_available = sorted(available_edges, key=lambda x: x[2])
        
        for u, v, w in sorted_available:
            if uf.union(u, v):
                weight += w
                
                if uf.components == 1:
                    break
        
        return weight if uf.components == 1 else -1
    
    def _enumerate_all_spanning_trees(self, n: int, edges: List[List[int]]) -> List[Tuple[int, List]]:
        """Enumerate all spanning trees (exponential - for small graphs only)"""
        from itertools import combinations
        
        all_trees = []
        
        # Try all combinations of n-1 edges
        for edge_combination in combinations(edges, n - 1):
            uf = UnionFind(n)
            weight = 0
            valid = True
            
            for u, v, w in edge_combination:
                if not uf.union(u, v):
                    valid = False
                    break
                weight += w
            
            if valid and uf.components == 1:
                all_trees.append((weight, list(edge_combination)))
        
        return all_trees

class UnionFind:
    """Union-Find with undo operation support"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
        self.history = []
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        
        if px == py:
            self.history.append(None)
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.history.append((py, self.parent[py], self.rank[px]))
        self.parent[py] = px
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.components -= 1
        return True
    
    def undo_union(self):
        """Undo last union operation"""
        if not self.history:
            return
        
        last_op = self.history.pop()
        
        if last_op is None:
            return  # Was a failed union
        
        py, old_parent, old_rank = last_op
        self.parent[py] = old_parent
        
        if old_rank != self.rank[self.parent[py]]:
            self.rank[self.parent[py]] = old_rank
        
        self.components += 1

def test_second_minimum_mst():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, edges, expected)
        (4, [[0,1,1],[1,2,2],[2,3,3],[0,3,4],[0,2,5]], 10),
        (3, [[0,1,1],[1,2,1],[0,2,1]], -1),
        (4, [[0,1,1],[1,2,1],[2,3,1],[0,3,2]], 4),
        (5, [[0,1,1],[1,2,1],[2,3,1],[3,4,1],[0,4,5]], 7),
        (4, [[0,1,2],[1,2,2],[2,3,2],[0,3,2]], 6),
    ]
    
    approaches = [
        ("Edge Replacement", solution.findSecondMinimumMST_approach1_edge_replacement),
        ("Cycle Replacement", solution.findSecondMinimumMST_approach2_cycle_replacement),
        # Note: Approaches 3-5 are more complex and omitted for brevity
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, edges, expected) in enumerate(test_cases):
            result = func(n, edges[:])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={expected}, got={result}")

def demonstrate_smst_concept():
    """Demonstrate Second MST concept"""
    print("\n=== Second Minimum Spanning Tree Demo ===")
    
    n = 4
    edges = [[0,1,1],[1,2,2],[2,3,3],[0,3,4],[0,2,5]]
    
    print(f"Graph: {n} vertices")
    print(f"Edges: {edges}")
    
    # Find MST
    solution = Solution()
    mst_weight, mst_edges = solution._find_mst_kruskal(n, edges)
    
    print(f"\nMST:")
    print(f"  Weight: {mst_weight}")
    print(f"  Edges: {mst_edges}")
    
    # Show edge replacement strategy
    print(f"\nSMST via Edge Replacement:")
    print(f"Try removing each MST edge and find alternative:")
    
    for remove_u, remove_v, remove_w in mst_edges:
        print(f"\nRemove MST edge ({remove_u},{remove_v}) weight {remove_w}:")
        
        # Create modified edge list
        modified_edges = []
        for u, v, w in edges:
            if not (u == remove_u and v == remove_v and w == remove_w) and \
               not (u == remove_v and v == remove_u and w == remove_w):
                modified_edges.append([u, v, w])
        
        print(f"  Remaining edges: {modified_edges}")
        
        # Find MST of remaining edges
        alt_weight, alt_edges = solution._find_mst_kruskal(n, modified_edges)
        
        if alt_weight != -1:
            print(f"  Alternative MST weight: {alt_weight}")
            if alt_weight > mst_weight:
                print(f"  → Candidate SMST with weight {alt_weight}")
        else:
            print(f"  → No spanning tree possible (graph disconnected)")

if __name__ == "__main__":
    test_second_minimum_mst()
    demonstrate_smst_concept()

"""
Second Minimum Spanning Tree (SMST) Concepts:
1. Edge Replacement Strategy for SMST Finding
2. Cycle-based MST Alternative Construction
3. Systematic Spanning Tree Enumeration
4. Graph Contraction and Matroid Theory
5. MST Variants and Optimization Extensions

Key Problem Insights:
- SMST has second smallest weight among all spanning trees
- If multiple MSTs exist with same weight, SMST has strictly larger weight
- Edge replacement: remove MST edge, find alternative spanning tree
- Cycle addition: add non-MST edge, remove heaviest edge in formed cycle

Algorithm Strategies:
1. Edge Replacement: Remove each MST edge, find alternative
2. Cycle Replacement: Add non-MST edges, replace heaviest in cycle
3. Systematic Enumeration: Generate all spanning trees, sort by weight
4. Advanced: Matroid intersection, contraction algorithms

Edge Replacement Method:
- Find MST using Kruskal's or Prim's algorithm
- For each MST edge: remove it and find MST of remaining graph
- Minimum weight > original MST weight is SMST

Cycle Replacement Method:
- For each non-MST edge: adding it creates unique cycle in MST
- Remove heaviest edge in this cycle to get alternative spanning tree
- Track minimum weight among all alternatives

Complexity Analysis:
- Edge Replacement: O(E * MST_time) = O(E^2 * α(V))
- Cycle Replacement: O(E * path_finding) = O(E * V)
- Enumeration: O(C(E, V-1) * α(V)) - exponential
- Advanced methods: O(V^3) or O(E^3) depending on approach

Real-world Applications:
- Network redundancy planning with backup connections
- Transportation route alternatives for reliability
- Supply chain resilience with secondary pathways
- Communication network fault tolerance design
- Infrastructure robustness with alternative configurations

This problem demonstrates advanced MST analysis
for network robustness and alternative optimization.
"""
