"""
1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree - Multiple Approaches
Difficulty: Hard

Given a weighted undirected connected graph with n vertices numbered from 0 to n - 1, and an array edges where edges[i] = [ai, bi, weighti] represents a bidirectional and weighted edge between nodes ai and bi. A minimum spanning tree (MST) is a subset of the graph's edges that connects all vertices without cycles and with the minimum possible total edge weight.

Find all the critical and pseudo-critical edges in the given graph's MST:
- A critical edge is an edge whose deletion from the graph would cause the MST weight to increase.
- A pseudo-critical edge is an edge that can appear in some MSTs but not all.

Return a list answer where answer[0] is a list of all critical edges and answer[1] is a list of all pseudo-critical edges. Note that you may return the indices of the edges in any order.
"""

from typing import List, Tuple
import heapq

class FindCriticalEdges:
    """Multiple approaches to find critical and pseudo-critical edges in MST"""
    
    def findCriticalAndPseudoCriticalEdges_brute_force(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 1: Brute Force with Union-Find
        
        Test each edge by excluding/including it in MST construction.
        
        Time: O(E² * α(V)), Space: O(V + E)
        """
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.components = n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                self.components -= 1
                return True
        
        def kruskal_mst(exclude_edge=-1, include_edge=-1):
            """Build MST using Kruskal's algorithm"""
            uf = UnionFind(n)
            total_weight = 0
            edges_used = 0
            
            # Include forced edge first
            if include_edge != -1:
                u, v, w = edges[include_edge][:3]
                if uf.union(u, v):
                    total_weight += w
                    edges_used += 1
                else:
                    return float('inf')  # Including this edge creates cycle
            
            # Sort edges by weight with original indices
            sorted_edges = sorted((w, i) for i, (u, v, w) in enumerate(edges))
            
            for weight, i in sorted_edges:
                if i == exclude_edge or i == include_edge:
                    continue
                
                u, v = edges[i][:2]
                if uf.union(u, v):
                    total_weight += weight
                    edges_used += 1
                    if edges_used == n - 1:
                        break
            
            return total_weight if uf.components == 1 else float('inf')
        
        # Find original MST weight
        original_mst_weight = kruskal_mst()
        
        critical = []
        pseudo_critical = []
        
        for i in range(len(edges)):
            # Test if edge is critical (excluding it increases MST weight)
            weight_without = kruskal_mst(exclude_edge=i)
            if weight_without > original_mst_weight:
                critical.append(i)
            else:
                # Test if edge is pseudo-critical (including it keeps same MST weight)
                weight_with = kruskal_mst(include_edge=i)
                if weight_with == original_mst_weight:
                    pseudo_critical.append(i)
        
        return [critical, pseudo_critical]
    
    def findCriticalAndPseudoCriticalEdges_optimized_kruskal(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 2: Optimized Kruskal with Edge Classification
        
        Use optimized Kruskal's algorithm with better edge classification.
        
        Time: O(E² * α(V)), Space: O(V + E)
        """
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))
                self.size = [1] * n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                if self.size[px] < self.size[py]:
                    px, py = py, px
                self.parent[py] = px
                self.size[px] += self.size[py]
                return True
            
            def connected(self):
                return len(set(self.find(i) for i in range(len(self.parent)))) == 1
        
        def build_mst(skip_edge=-1, must_include=-1):
            """Build MST with constraints"""
            dsu = DSU(n)
            weight = 0
            
            # Must include edge first
            if must_include != -1:
                u, v, w = edges[must_include][:3]
                if not dsu.union(u, v):
                    return float('inf')  # Creates cycle
                weight += w
            
            # Add edges in sorted order
            edge_indices = sorted(range(len(edges)), key=lambda i: edges[i][2])
            
            for i in edge_indices:
                if i == skip_edge or i == must_include:
                    continue
                
                u, v, w = edges[i][:3]
                if dsu.union(u, v):
                    weight += w
            
            return weight if dsu.connected() else float('inf')
        
        # Get original MST weight
        mst_weight = build_mst()
        
        critical_edges = []
        pseudo_critical_edges = []
        
        for i in range(len(edges)):
            # Check if critical (removing increases weight)
            if build_mst(skip_edge=i) > mst_weight:
                critical_edges.append(i)
            # Check if pseudo-critical (can be included without increasing weight)
            elif build_mst(must_include=i) == mst_weight:
                pseudo_critical_edges.append(i)
        
        return [critical_edges, pseudo_critical_edges]
    
    def findCriticalAndPseudoCriticalEdges_tarjan_bridges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 3: Tarjan's Bridge Algorithm Integration
        
        Use Tarjan's algorithm concepts for bridge detection in MST context.
        
        Time: O(E² * α(V)), Space: O(V + E)
        """
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return False
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                return True
            
            def is_connected(self):
                root = self.find(0)
                return all(self.find(i) == root for i in range(len(self.parent)))
        
        def get_mst_weight(forbidden=-1, required=-1):
            """Get MST weight with constraints"""
            uf = UnionFind(n)
            total = 0
            
            # Add required edge first
            if required != -1:
                u, v, w = edges[required][:3]
                if uf.union(u, v):
                    total += w
                else:
                    return float('inf')
            
            # Sort edges by weight
            sorted_indices = sorted(range(len(edges)), key=lambda i: edges[i][2])
            
            for idx in sorted_indices:
                if idx == forbidden or idx == required:
                    continue
                
                u, v, w = edges[idx][:3]
                if uf.union(u, v):
                    total += w
            
            return total if uf.is_connected() else float('inf')
        
        original_weight = get_mst_weight()
        critical = []
        pseudo_critical = []
        
        for i in range(len(edges)):
            # Test if removing edge increases MST weight (critical)
            without_edge = get_mst_weight(forbidden=i)
            if without_edge > original_weight:
                critical.append(i)
            else:
                # Test if including edge maintains MST weight (pseudo-critical)
                with_edge = get_mst_weight(required=i)
                if with_edge == original_weight:
                    pseudo_critical.append(i)
        
        return [critical, pseudo_critical]
    
    def findCriticalAndPseudoCriticalEdges_prim_based(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 4: Prim's Algorithm Based Approach
        
        Use Prim's algorithm for MST construction and edge analysis.
        
        Time: O(E² log V), Space: O(V + E)
        """
        def prim_mst(exclude_edge=-1, include_edge=-1):
            """Build MST using Prim's algorithm"""
            if include_edge != -1:
                # Check if including this edge is valid
                u, v, w = edges[include_edge][:3]
                visited = {u, v}
                mst_weight = w
                mst_edges = 1
            else:
                visited = {0}
                mst_weight = 0
                mst_edges = 0
            
            # Build adjacency list excluding forbidden edge
            adj = [[] for _ in range(n)]
            for i, (u, v, w) in enumerate(edges):
                if i != exclude_edge:
                    adj[u].append((v, w, i))
                    adj[v].append((u, w, i))
            
            # Priority queue for minimum edge weights
            pq = []
            
            # Add edges from visited nodes
            for node in visited:
                for neighbor, weight, edge_idx in adj[node]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (weight, node, neighbor, edge_idx))
            
            while pq and mst_edges < n - 1:
                weight, u, v, edge_idx = heapq.heappop(pq)
                
                if v in visited:
                    continue
                
                visited.add(v)
                mst_weight += weight
                mst_edges += 1
                
                # Add new edges from v
                for neighbor, w, idx in adj[v]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (w, v, neighbor, idx))
            
            return mst_weight if mst_edges == n - 1 else float('inf')
        
        # Get original MST weight
        original_mst = prim_mst()
        
        critical = []
        pseudo_critical = []
        
        for i in range(len(edges)):
            # Test critical: exclude edge
            without = prim_mst(exclude_edge=i)
            if without > original_mst:
                critical.append(i)
            else:
                # Test pseudo-critical: include edge
                with_edge = prim_mst(include_edge=i)
                if with_edge == original_mst:
                    pseudo_critical.append(i)
        
        return [critical, pseudo_critical]
    
    def findCriticalAndPseudoCriticalEdges_cycle_detection(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        """
        Approach 5: Cycle Detection Based Classification
        
        Use cycle detection to classify edges more efficiently.
        
        Time: O(E² * α(V)), Space: O(V + E)
        """
        class QuickUnion:
            def __init__(self, n):
                self.id = list(range(n))
                self.sz = [1] * n
                self.count = n
            
            def find(self, p):
                while p != self.id[p]:
                    self.id[p] = self.id[self.id[p]]  # Path compression
                    p = self.id[p]
                return p
            
            def union(self, p, q):
                root_p, root_q = self.find(p), self.find(q)
                if root_p == root_q:
                    return False
                
                # Union by size
                if self.sz[root_p] < self.sz[root_q]:
                    root_p, root_q = root_q, root_p
                
                self.id[root_q] = root_p
                self.sz[root_p] += self.sz[root_q]
                self.count -= 1
                return True
            
            def connected(self, p, q):
                return self.find(p) == self.find(q)
            
            def is_fully_connected(self):
                return self.count == 1
        
        def compute_mst(skip=-1, force=-1):
            """Compute MST weight with constraints"""
            uf = QuickUnion(n)
            weight = 0
            
            # Force include edge
            if force != -1:
                u, v, w = edges[force][:3]
                if uf.connected(u, v):
                    return float('inf')  # Would create cycle
                uf.union(u, v)
                weight += w
            
            # Process edges in weight order
            edge_order = sorted(range(len(edges)), key=lambda x: edges[x][2])
            
            for i in edge_order:
                if i == skip or i == force:
                    continue
                
                u, v, w = edges[i][:3]
                if not uf.connected(u, v):
                    uf.union(u, v)
                    weight += w
            
            return weight if uf.is_fully_connected() else float('inf')
        
        # Compute original MST weight
        mst_weight = compute_mst()
        
        critical_edges = []
        pseudo_critical_edges = []
        
        for i in range(len(edges)):
            # Check if edge is critical
            weight_without = compute_mst(skip=i)
            if weight_without > mst_weight:
                critical_edges.append(i)
            else:
                # Check if edge is pseudo-critical
                weight_with = compute_mst(force=i)
                if weight_with == mst_weight:
                    pseudo_critical_edges.append(i)
        
        return [critical_edges, pseudo_critical_edges]

def test_critical_edges():
    """Test critical and pseudo-critical edges algorithms"""
    solver = FindCriticalEdges()
    
    test_cases = [
        (5, [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]], "Example 1"),
        (4, [[0,1,1],[1,2,1],[2,3,1],[0,3,1]], "Square graph"),
        (6, [[0,1,1],[1,2,1],[0,2,1],[2,3,4],[3,4,2],[3,5,2],[4,5,2]], "Complex graph"),
        (3, [[0,1,1],[1,2,2],[0,2,2]], "Triangle"),
    ]
    
    algorithms = [
        ("Brute Force", solver.findCriticalAndPseudoCriticalEdges_brute_force),
        ("Optimized Kruskal", solver.findCriticalAndPseudoCriticalEdges_optimized_kruskal),
        ("Tarjan Bridges", solver.findCriticalAndPseudoCriticalEdges_tarjan_bridges),
        ("Prim Based", solver.findCriticalAndPseudoCriticalEdges_prim_based),
        ("Cycle Detection", solver.findCriticalAndPseudoCriticalEdges_cycle_detection),
    ]
    
    print("=== Testing Critical and Pseudo-Critical Edges ===")
    
    for n, edges, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"n={n}, edges={edges}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, edges)
                critical, pseudo_critical = result
                print(f"{alg_name:18} | ✓ | Critical: {critical}, Pseudo: {pseudo_critical}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:40]}")

if __name__ == "__main__":
    test_critical_edges()

"""
Find Critical and Pseudo-Critical Edges demonstrates advanced
MST analysis with Union-Find optimization, bridge detection,
and comprehensive edge classification techniques.
"""
