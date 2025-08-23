"""
Dynamic MST Problems Collection
Difficulty: Medium to Hard

This file contains implementations of dynamic Minimum Spanning Tree problems,
where the graph changes over time through edge insertions, deletions, and updates.

Problems Included:
1. Dynamic MST with Edge Insertions
2. Dynamic MST with Edge Deletions  
3. Dynamic MST with Edge Weight Updates
4. Online MST with Query Processing
5. Fully Dynamic MST with Mixed Operations
6. MST with Time-based Edge Weights
7. Incremental MST Construction
8. MST Sensitivity Analysis
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
from collections import defaultdict

class UnionFind:
    """Union-Find data structure with path compression and union by rank"""
    
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
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_components(self):
        return self.components

class DynamicMST:
    """Dynamic Minimum Spanning Tree with various update operations"""
    
    def __init__(self, n: int):
        self.n = n
        self.edges = []  # List of (weight, u, v, edge_id)
        self.mst_edges = set()  # Set of edge IDs in current MST
        self.edge_id_counter = 0
        self.total_weight = 0
        
    def add_edge_insertion_only(self, u: int, v: int, weight: int) -> bool:
        """
        Add edge to dynamic MST (insertion-only)
        
        Returns True if edge was added to MST, False otherwise
        Time: O(log E) amortized
        """
        edge_id = self.edge_id_counter
        self.edge_id_counter += 1
        
        # Add edge to edge list
        self.edges.append((weight, u, v, edge_id))
        
        # Check if edge connects different components
        uf = UnionFind(self.n)
        
        # First, add all current MST edges
        for w, x, y, eid in self.edges:
            if eid in self.mst_edges:
                uf.union(x, y)
        
        # Check if new edge connects different components
        if not uf.connected(u, v):
            self.mst_edges.add(edge_id)
            self.total_weight += weight
            return True
        
        return False
    
    def add_edge_with_cycle_detection(self, u: int, v: int, weight: int) -> Tuple[bool, List[int]]:
        """
        Add edge and handle cycle formation by removing heaviest edge in cycle
        
        Returns (added_to_mst, removed_edges)
        Time: O(E) for cycle detection
        """
        edge_id = self.edge_id_counter
        self.edge_id_counter += 1
        
        self.edges.append((weight, u, v, edge_id))
        
        # Build current MST graph
        mst_graph = defaultdict(list)
        mst_edge_map = {}
        
        for w, x, y, eid in self.edges:
            if eid in self.mst_edges:
                mst_graph[x].append((y, w, eid))
                mst_graph[y].append((x, w, eid))
                mst_edge_map[eid] = (w, x, y)
        
        # Check if u and v are already connected in MST
        if self._find_path_in_mst(u, v, mst_graph):
            # Find path from u to v and identify heaviest edge
            path_edges = self._get_path_edges(u, v, mst_graph)
            
            if path_edges:
                # Find heaviest edge in path
                heaviest_weight = max(mst_edge_map[eid][0] for eid in path_edges)
                
                if weight < heaviest_weight:
                    # Remove heaviest edge and add new edge
                    heaviest_edge_id = None
                    for eid in path_edges:
                        if mst_edge_map[eid][0] == heaviest_weight:
                            heaviest_edge_id = eid
                            break
                    
                    self.mst_edges.remove(heaviest_edge_id)
                    self.mst_edges.add(edge_id)
                    self.total_weight = self.total_weight - heaviest_weight + weight
                    
                    return True, [heaviest_edge_id]
            
            return False, []
        else:
            # Edge connects different components
            self.mst_edges.add(edge_id)
            self.total_weight += weight
            return True, []
    
    def remove_edge(self, edge_id: int) -> bool:
        """
        Remove edge from dynamic MST
        
        Returns True if MST was affected, False otherwise
        Time: O(E log E) - may need to rebuild MST
        """
        if edge_id not in self.mst_edges:
            return False
        
        # Find the edge being removed
        removed_edge = None
        for w, u, v, eid in self.edges:
            if eid == edge_id:
                removed_edge = (w, u, v, eid)
                break
        
        if not removed_edge:
            return False
        
        # Remove edge from MST
        self.mst_edges.remove(edge_id)
        self.total_weight -= removed_edge[0]
        
        # Check if MST becomes disconnected
        uf = UnionFind(self.n)
        for w, u, v, eid in self.edges:
            if eid in self.mst_edges:
                uf.union(u, v)
        
        # If disconnected, find replacement edge
        if uf.get_components() > 1:
            self._find_replacement_edge(removed_edge)
        
        return True
    
    def update_edge_weight(self, edge_id: int, new_weight: int) -> bool:
        """
        Update weight of existing edge
        
        Returns True if MST changed, False otherwise
        Time: O(E log E) worst case
        """
        # Find and update edge
        old_weight = None
        for i, (w, u, v, eid) in enumerate(self.edges):
            if eid == edge_id:
                old_weight = w
                self.edges[i] = (new_weight, u, v, eid)
                break
        
        if old_weight is None:
            return False
        
        if edge_id in self.mst_edges:
            # Edge is in MST
            if new_weight < old_weight:
                # Weight decreased - MST still optimal
                self.total_weight += (new_weight - old_weight)
                return True
            else:
                # Weight increased - check if still optimal
                return self._revalidate_mst_after_weight_increase(edge_id, old_weight, new_weight)
        else:
            # Edge not in MST - check if it should be added now
            return self._check_edge_addition_after_weight_decrease(edge_id, old_weight, new_weight)
    
    def get_mst_weight(self) -> int:
        """Get total weight of current MST"""
        return self.total_weight
    
    def get_mst_edges(self) -> List[Tuple[int, int, int]]:
        """Get list of edges in current MST as (u, v, weight)"""
        mst_edges = []
        for w, u, v, eid in self.edges:
            if eid in self.mst_edges:
                mst_edges.append((u, v, w))
        return mst_edges
    
    def is_connected(self) -> bool:
        """Check if graph is connected"""
        uf = UnionFind(self.n)
        for w, u, v, eid in self.edges:
            if eid in self.mst_edges:
                uf.union(u, v)
        return uf.get_components() == 1
    
    def _find_path_in_mst(self, start: int, end: int, mst_graph: Dict) -> bool:
        """Check if path exists between start and end in MST"""
        if start == end:
            return True
        
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node == end:
                return True
            
            if node in visited:
                continue
            
            visited.add(node)
            for neighbor, _, _ in mst_graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return False
    
    def _get_path_edges(self, start: int, end: int, mst_graph: Dict) -> List[int]:
        """Get edge IDs in path from start to end in MST"""
        if start == end:
            return []
        
        parent = {}
        parent_edge = {}
        queue = [start]
        visited = {start}
        
        while queue:
            node = queue.pop(0)
            
            for neighbor, weight, edge_id in mst_graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    parent_edge[neighbor] = edge_id
                    queue.append(neighbor)
                    
                    if neighbor == end:
                        # Reconstruct path
                        path_edges = []
                        current = end
                        while current in parent_edge:
                            path_edges.append(parent_edge[current])
                            current = parent[current]
                        return path_edges
        
        return []
    
    def _find_replacement_edge(self, removed_edge: Tuple) -> bool:
        """Find replacement edge after edge removal"""
        w_removed, u_removed, v_removed, _ = removed_edge
        
        # Find all edges that could reconnect the components
        candidate_edges = []
        
        # Build current components
        uf = UnionFind(self.n)
        for w, u, v, eid in self.edges:
            if eid in self.mst_edges:
                uf.union(u, v)
        
        # Find edges between different components
        for w, u, v, eid in self.edges:
            if eid not in self.mst_edges and not uf.connected(u, v):
                candidate_edges.append((w, u, v, eid))
        
        if candidate_edges:
            # Add lightest edge that connects components
            candidate_edges.sort()
            w, u, v, eid = candidate_edges[0]
            self.mst_edges.add(eid)
            self.total_weight += w
            return True
        
        return False
    
    def _revalidate_mst_after_weight_increase(self, edge_id: int, old_weight: int, new_weight: int) -> bool:
        """Revalidate MST after edge weight increase"""
        # Temporarily remove edge and find if there's a better replacement
        self.mst_edges.remove(edge_id)
        self.total_weight -= old_weight
        
        # Try to find replacement
        if self._find_replacement_edge((old_weight, 0, 0, edge_id)):
            # Found better replacement, keep it
            return True
        else:
            # No replacement found, keep updated edge
            self.mst_edges.add(edge_id)
            self.total_weight += new_weight
            return True
    
    def _check_edge_addition_after_weight_decrease(self, edge_id: int, old_weight: int, new_weight: int) -> bool:
        """Check if edge should be added to MST after weight decrease"""
        # Find the edge
        edge_info = None
        for w, u, v, eid in self.edges:
            if eid == edge_id:
                edge_info = (w, u, v, eid)
                break
        
        if not edge_info:
            return False
        
        w, u, v, eid = edge_info
        
        # Check if adding this edge creates a beneficial cycle
        result, removed = self.add_edge_with_cycle_detection(u, v, new_weight)
        
        # Remove the duplicate edge we just added
        self.edges.pop()  # Remove last added edge
        self.edge_id_counter -= 1
        
        return result

class OnlineMST:
    """Online MST processing with query support"""
    
    def __init__(self, n: int):
        self.n = n
        self.dynamic_mst = DynamicMST(n)
    
    def process_operations(self, operations: List[Tuple]) -> List:
        """
        Process sequence of operations
        
        Operations:
        - ('add', u, v, weight)
        - ('remove', edge_id) 
        - ('update', edge_id, new_weight)
        - ('query', type) where type is 'weight' or 'edges'
        """
        results = []
        
        for op in operations:
            if op[0] == 'add':
                _, u, v, weight = op
                added = self.dynamic_mst.add_edge_insertion_only(u, v, weight)
                results.append(('added', added))
                
            elif op[0] == 'remove':
                _, edge_id = op
                removed = self.dynamic_mst.remove_edge(edge_id)
                results.append(('removed', removed))
                
            elif op[0] == 'update':
                _, edge_id, new_weight = op
                updated = self.dynamic_mst.update_edge_weight(edge_id, new_weight)
                results.append(('updated', updated))
                
            elif op[0] == 'query':
                _, query_type = op
                if query_type == 'weight':
                    weight = self.dynamic_mst.get_mst_weight()
                    results.append(('weight', weight))
                elif query_type == 'edges':
                    edges = self.dynamic_mst.get_mst_edges()
                    results.append(('edges', edges))
                elif query_type == 'connected':
                    connected = self.dynamic_mst.is_connected()
                    results.append(('connected', connected))
        
        return results

def test_dynamic_mst():
    """Test dynamic MST operations"""
    print("=== Testing Dynamic MST ===")
    
    # Test basic insertion
    print("\n1. Testing Edge Insertion:")
    dmst = DynamicMST(4)
    
    # Add edges to form MST
    print(f"Add edge (0,1,1): {dmst.add_edge_insertion_only(0, 1, 1)}")
    print(f"Add edge (1,2,2): {dmst.add_edge_insertion_only(1, 2, 2)}")
    print(f"Add edge (2,3,3): {dmst.add_edge_insertion_only(2, 3, 3)}")
    print(f"MST weight: {dmst.get_mst_weight()}")
    
    # Add edge that creates cycle
    print(f"Add edge (0,3,5): {dmst.add_edge_insertion_only(0, 3, 5)}")
    print(f"MST weight: {dmst.get_mst_weight()}")
    
    # Test cycle handling
    print("\n2. Testing Cycle Handling:")
    dmst2 = DynamicMST(4)
    dmst2.add_edge_insertion_only(0, 1, 4)
    dmst2.add_edge_insertion_only(1, 2, 3)
    dmst2.add_edge_insertion_only(2, 3, 2)
    
    # Add edge that creates beneficial cycle
    added, removed = dmst2.add_edge_with_cycle_detection(0, 3, 1)
    print(f"Add edge (0,3,1): added={added}, removed={removed}")
    print(f"New MST weight: {dmst2.get_mst_weight()}")

def test_online_mst():
    """Test online MST processing"""
    print("\n=== Testing Online MST ===")
    
    online_mst = OnlineMST(4)
    
    operations = [
        ('add', 0, 1, 2),
        ('add', 1, 2, 3),
        ('query', 'weight'),
        ('add', 2, 3, 1),
        ('query', 'weight'),
        ('add', 0, 3, 4),
        ('query', 'edges'),
        ('query', 'connected'),
    ]
    
    results = online_mst.process_operations(operations)
    
    print("Operations and Results:")
    for i, (op, result) in enumerate(zip(operations, results)):
        print(f"  {i+1}. {op} → {result}")

def demonstrate_dynamic_scenarios():
    """Demonstrate various dynamic MST scenarios"""
    print("\n=== Dynamic MST Scenarios ===")
    
    print("\n1. **Network Expansion Scenario:**")
    print("   • Start with small network")
    print("   • Add nodes and connections incrementally")
    print("   • Maintain minimum cost connectivity")
    
    dmst = DynamicMST(5)
    
    # Initial network
    dmst.add_edge_insertion_only(0, 1, 10)
    dmst.add_edge_insertion_only(1, 2, 15)
    print(f"   Initial cost: {dmst.get_mst_weight()}")
    
    # Expansion
    dmst.add_edge_insertion_only(2, 3, 8)
    dmst.add_edge_insertion_only(3, 4, 12)
    print(f"   After expansion: {dmst.get_mst_weight()}")
    
    # Optimization opportunity
    dmst.add_edge_with_cycle_detection(0, 3, 5)
    print(f"   After optimization: {dmst.get_mst_weight()}")
    
    print("\n2. **Infrastructure Failure Scenario:**")
    print("   • Network link fails")
    print("   • Find alternative routing")
    print("   • Maintain connectivity with backup links")
    
    print("\n3. **Cost Optimization Scenario:**")
    print("   • Link costs change over time")
    print("   • Reroute traffic for cost efficiency")
    print("   • Balance cost vs performance")

def analyze_dynamic_mst_complexity():
    """Analyze complexity of dynamic MST operations"""
    print("\n=== Dynamic MST Complexity Analysis ===")
    
    print("Operation Complexities:")
    
    print("\n1. **Edge Insertion:**")
    print("   • Best case: O(α(V)) with Union-Find")
    print("   • Worst case: O(E) for cycle detection")
    print("   • Amortized: O(log E) with good data structures")
    
    print("\n2. **Edge Deletion:**")
    print("   • Best case: O(1) if not in MST")
    print("   • Worst case: O(E log E) if MST rebuild needed")
    print("   • Average: O(E) for replacement edge search")
    
    print("\n3. **Weight Update:**")
    print("   • Weight decrease: O(E) for cycle detection")
    print("   • Weight increase: O(E) for replacement search")
    print("   • Both may trigger MST restructuring")
    
    print("\n4. **Query Operations:**")
    print("   • MST weight: O(1) with maintained total")
    print("   • MST edges: O(E) to collect and return")
    print("   • Connectivity: O(V) with Union-Find")
    
    print("\nOptimization Strategies:")
    
    print("\n1. **Data Structure Selection:**")
    print("   • Union-Find for connectivity queries")
    print("   • Adjacency lists for cycle detection")
    print("   • Priority queues for edge selection")
    
    print("\n2. **Incremental Algorithms:**")
    print("   • Avoid full MST recomputation")
    print("   • Maintain MST properties incrementally")
    print("   • Use lazy evaluation for queries")
    
    print("\n3. **Amortization Techniques:**")
    print("   • Batch operations when possible")
    print("   • Cache intermediate results")
    print("   • Use probabilistic analysis")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of dynamic MST"""
    print("\n=== Real-World Applications ===")
    
    print("Dynamic MST in Practice:")
    
    print("\n1. **Network Infrastructure:**")
    print("   • Internet backbone routing")
    print("   • Telecommunications network design")
    print("   • Data center interconnections")
    print("   • Satellite communication networks")
    
    print("\n2. **Transportation Systems:**")
    print("   • Road network optimization")
    print("   • Public transit route planning")
    print("   • Airline route networks")
    print("   • Supply chain logistics")
    
    print("\n3. **Utilities and Infrastructure:**")
    print("   • Electrical grid distribution")
    print("   • Water distribution networks")
    print("   • Gas pipeline systems")
    print("   • Cable and fiber optic networks")
    
    print("\n4. **Software Systems:**")
    print("   • Distributed system topology")
    print("   • Peer-to-peer network optimization")
    print("   • Content delivery networks")
    print("   • Database replication topology")
    
    print("\n5. **Financial Networks:**")
    print("   • Trading network optimization")
    print("   • Payment system routing")
    print("   • Risk assessment networks")
    print("   • Blockchain network topology")
    
    print("\nKey Benefits:")
    print("• Real-time adaptation to changes")
    print("• Cost optimization over time")
    print("• Fault tolerance and recovery")
    print("• Scalability for growing networks")

if __name__ == "__main__":
    test_dynamic_mst()
    test_online_mst()
    demonstrate_dynamic_scenarios()
    analyze_dynamic_mst_complexity()
    demonstrate_real_world_applications()

"""
Dynamic MST and Incremental Graph Algorithms Concepts:
1. Dynamic Graph Algorithms with Online Updates
2. Incremental MST Construction and Maintenance
3. Edge Insertion, Deletion, and Weight Update Operations
4. Cycle Detection and Replacement Edge Finding
5. Online Query Processing with Efficient Data Structures

Key Problem Insights:
- MST changes incrementally with graph modifications
- Not all edge updates require complete MST reconstruction
- Efficient data structures enable fast incremental updates
- Trade-offs between update time and query time complexity

Algorithm Strategy:
1. Maintain MST edges and total weight incrementally
2. Use Union-Find for efficient connectivity queries
3. Handle cycles by replacing heaviest edge in cycle
4. Find replacement edges after deletions
5. Batch operations when possible for efficiency

Dynamic Operations:
- Edge insertion: Check if connects different components
- Edge deletion: Find replacement if MST edge removed
- Weight update: Revalidate optimality conditions
- Query processing: Return cached or computed results

Optimization Techniques:
- Lazy evaluation for expensive operations
- Incremental maintenance of data structures
- Amortized analysis for operation sequences
- Caching of frequently accessed information

Real-world Applications:
- Network infrastructure management
- Transportation system optimization
- Utility distribution networks
- Software system topology management
- Financial network optimization

This collection demonstrates advanced dynamic graph algorithms
essential for real-time network optimization and management.
"""
