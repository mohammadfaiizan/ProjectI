"""
MST with Constraints Collection
Difficulty: Medium-Hard

This file contains various MST problems with additional constraints that demonstrate
advanced MST algorithms and optimization techniques.

Problems Included:
1. MST with Budget Constraint
2. MST with Degree Constraint
3. MST with Color Constraint
4. MST with Distance Constraint
5. Dynamic MST with Edge Updates
6. MST with Forbidden Edges
7. MST with Priority Vertices
8. Multi-Objective MST Optimization
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq
from collections import defaultdict

class MSTWithConstraints:
    """Collection of MST algorithms with various constraints"""
    
    def mst_with_budget_constraint(self, n: int, edges: List[List[int]], budget: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST with Budget Constraint
        
        Find MST with total cost <= budget, or maximum forest within budget.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        uf = UnionFind(n)
        total_cost = 0
        mst_edges = []
        
        for u, v, cost in sorted_edges:
            if total_cost + cost <= budget and uf.union(u, v):
                total_cost += cost
                mst_edges.append((u, v))
                
                # Early termination if we have spanning tree
                if len(mst_edges) == n - 1:
                    break
        
        return total_cost, mst_edges
    
    def mst_with_degree_constraint(self, n: int, edges: List[List[int]], max_degree: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST with Degree Constraint
        
        Find MST where no vertex has degree > max_degree.
        
        Time: O(E log E + E * V)
        Space: O(V + E)
        """
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        uf = UnionFind(n)
        degree = [0] * n
        total_cost = 0
        mst_edges = []
        
        for u, v, cost in sorted_edges:
            # Check degree constraint
            if (degree[u] < max_degree and degree[v] < max_degree and 
                uf.union(u, v)):
                
                total_cost += cost
                mst_edges.append((u, v))
                degree[u] += 1
                degree[v] += 1
                
                if len(mst_edges) == n - 1:
                    break
        
        return total_cost, mst_edges
    
    def mst_with_color_constraint(self, n: int, edges: List[List[int]], 
                                 vertex_colors: List[int], max_color_edges: Dict[int, int]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST with Color Constraint
        
        Limit number of edges between vertices of specific colors.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        uf = UnionFind(n)
        color_edge_count = defaultdict(int)
        total_cost = 0
        mst_edges = []
        
        for u, v, cost in sorted_edges:
            color_u, color_v = vertex_colors[u], vertex_colors[v]
            
            # Check color constraint
            if color_u == color_v:
                edge_type = color_u
                if color_edge_count[edge_type] >= max_color_edges.get(edge_type, float('inf')):
                    continue
            
            if uf.union(u, v):
                total_cost += cost
                mst_edges.append((u, v))
                
                if color_u == color_v:
                    color_edge_count[color_u] += 1
                
                if len(mst_edges) == n - 1:
                    break
        
        return total_cost, mst_edges
    
    def mst_with_distance_constraint(self, n: int, edges: List[List[int]], 
                                   coordinates: List[Tuple[float, float]], 
                                   max_distance: float) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST with Distance Constraint
        
        Only consider edges where Euclidean distance <= max_distance.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        import math
        
        # Filter edges by distance constraint
        valid_edges = []
        
        for u, v, cost in edges:
            x1, y1 = coordinates[u]
            x2, y2 = coordinates[v]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            if distance <= max_distance:
                valid_edges.append([u, v, cost])
        
        # Standard MST on valid edges
        return self._standard_mst(n, valid_edges)
    
    def dynamic_mst_with_updates(self, n: int, initial_edges: List[List[int]], 
                               updates: List[Tuple[str, int, int, int]]) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """
        Dynamic MST with Edge Updates
        
        Handle sequence of edge additions and deletions.
        
        Time: O(U * E log E) for U updates
        Space: O(V + E)
        """
        current_edges = initial_edges[:]
        results = []
        
        # Initial MST
        initial_cost, initial_mst = self._standard_mst(n, current_edges)
        results.append((initial_cost, initial_mst))
        
        for operation, u, v, cost in updates:
            if operation == "add":
                current_edges.append([u, v, cost])
            elif operation == "remove":
                current_edges = [e for e in current_edges 
                               if not (e[0] == u and e[1] == v and e[2] == cost)]
            
            # Recompute MST after update
            new_cost, new_mst = self._standard_mst(n, current_edges)
            results.append((new_cost, new_mst))
        
        return results
    
    def mst_with_forbidden_edges(self, n: int, edges: List[List[int]], 
                               forbidden: Set[Tuple[int, int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST with Forbidden Edges
        
        Find MST while avoiding specified forbidden edges.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        # Filter out forbidden edges
        allowed_edges = []
        
        for u, v, cost in edges:
            edge_key = (min(u, v), max(u, v))
            if edge_key not in forbidden:
                allowed_edges.append([u, v, cost])
        
        return self._standard_mst(n, allowed_edges)
    
    def mst_with_priority_vertices(self, n: int, edges: List[List[int]], 
                                  priority_vertices: Set[int], 
                                  priority_bonus: float) -> Tuple[int, List[Tuple[int, int]]]:
        """
        MST with Priority Vertices
        
        Give preference to edges connecting to priority vertices.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        # Modify edge weights based on priority
        modified_edges = []
        
        for u, v, cost in edges:
            if u in priority_vertices or v in priority_vertices:
                # Apply bonus (reduce cost for priority connections)
                modified_cost = cost * (1 - priority_bonus)
            else:
                modified_cost = cost
            
            modified_edges.append([u, v, modified_cost])
        
        return self._standard_mst(n, modified_edges)
    
    def multi_objective_mst(self, n: int, edges: List[List[int]], 
                           objectives: List[str], weights: List[float]) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Multi-Objective MST Optimization
        
        Optimize multiple objectives simultaneously with weighted sum.
        
        Time: O(E log E + E * α(V))
        Space: O(V + E)
        """
        # Calculate composite scores for edges
        composite_edges = []
        
        for u, v, *obj_values in edges:
            composite_score = sum(w * val for w, val in zip(weights, obj_values))
            composite_edges.append([u, v, composite_score])
        
        return self._standard_mst(n, composite_edges)
    
    def capacitated_mst(self, n: int, edges: List[List[int]], 
                       capacities: List[int], demand: List[int], 
                       root: int) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Capacitated MST (CMST)
        
        MST where each subtree's demand doesn't exceed edge capacities.
        
        Time: O(E^2 * α(V)) - approximate algorithm
        Space: O(V + E)
        """
        # Simplified capacitated MST using approximation
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        uf = UnionFind(n)
        total_cost = 0
        mst_edges = []
        subtree_demand = demand[:]
        
        for u, v, cost in sorted_edges:
            if uf.find(u) != uf.find(v):
                # Check capacity constraint
                root_u, root_v = uf.find(u), uf.find(v)
                combined_demand = subtree_demand[root_u] + subtree_demand[root_v]
                
                # Find minimum capacity on path to root
                min_capacity = min(capacities[u], capacities[v])
                
                if combined_demand <= min_capacity and uf.union(u, v):
                    total_cost += cost
                    mst_edges.append((u, v))
                    
                    # Update demand for new root
                    new_root = uf.find(u)
                    subtree_demand[new_root] = combined_demand
                    
                    if len(mst_edges) == n - 1:
                        break
        
        return total_cost, mst_edges
    
    def _standard_mst(self, n: int, edges: List[List[int]]) -> Tuple[int, List[Tuple[int, int]]]:
        """Standard MST using Kruskal's algorithm"""
        if not edges:
            return 0, []
        
        sorted_edges = sorted(edges, key=lambda x: x[2])
        uf = UnionFind(n)
        
        total_cost = 0
        mst_edges = []
        
        for u, v, cost in sorted_edges:
            if uf.union(u, v):
                total_cost += cost
                mst_edges.append((u, v))
                
                if len(mst_edges) == n - 1:
                    break
        
        return total_cost, mst_edges

class UnionFind:
    """Optimized Union-Find data structure"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
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

def test_constrained_mst():
    """Test constrained MST algorithms"""
    mst = MSTWithConstraints()
    
    print("=== Testing Constrained MST Algorithms ===")
    
    # Test graph
    n = 5
    edges = [[0,1,2],[1,2,3],[2,3,1],[3,4,4],[0,4,5],[1,3,2]]
    
    print(f"\nTest Graph: {n} vertices, edges: {edges}")
    
    # Test 1: Budget constraint
    budget = 8
    cost, mst_edges = mst.mst_with_budget_constraint(n, edges, budget)
    print(f"\nBudget Constraint (budget={budget}):")
    print(f"  Cost: {cost}, Edges: {mst_edges}")
    
    # Test 2: Degree constraint
    max_degree = 2
    cost, mst_edges = mst.mst_with_degree_constraint(n, edges, max_degree)
    print(f"\nDegree Constraint (max_degree={max_degree}):")
    print(f"  Cost: {cost}, Edges: {mst_edges}")
    
    # Test 3: Color constraint
    colors = [0, 0, 1, 1, 2]
    color_limits = {0: 1, 1: 1, 2: 2}
    cost, mst_edges = mst.mst_with_color_constraint(n, edges, colors, color_limits)
    print(f"\nColor Constraint (colors={colors}, limits={color_limits}):")
    print(f"  Cost: {cost}, Edges: {mst_edges}")
    
    # Test 4: Forbidden edges
    forbidden = {(0, 4), (1, 3)}
    cost, mst_edges = mst.mst_with_forbidden_edges(n, edges, forbidden)
    print(f"\nForbidden Edges (forbidden={forbidden}):")
    print(f"  Cost: {cost}, Edges: {mst_edges}")

def demonstrate_constraint_impact():
    """Demonstrate impact of different constraints on MST"""
    print("\n=== Constraint Impact Analysis ===")
    
    mst = MSTWithConstraints()
    n = 4
    edges = [[0,1,1],[1,2,2],[2,3,3],[0,3,4],[0,2,5]]
    
    print(f"Base graph: {edges}")
    
    # Standard MST
    cost, mst_edges = mst._standard_mst(n, edges)
    print(f"\nStandard MST:")
    print(f"  Cost: {cost}, Edges: {mst_edges}")
    
    # Different budget constraints
    budgets = [6, 7, 8, 10]
    print(f"\nBudget Constraint Impact:")
    
    for budget in budgets:
        cost, mst_edges = mst.mst_with_budget_constraint(n, edges, budget)
        print(f"  Budget {budget}: Cost {cost}, Edges {len(mst_edges)}")
    
    # Different degree constraints
    degrees = [1, 2, 3]
    print(f"\nDegree Constraint Impact:")
    
    for max_deg in degrees:
        cost, mst_edges = mst.mst_with_degree_constraint(n, edges, max_deg)
        print(f"  Max degree {max_deg}: Cost {cost}, Edges {len(mst_edges)}")

def analyze_constraint_types():
    """Analyze different types of MST constraints"""
    print("\n=== MST Constraint Types Analysis ===")
    
    print("Constraint Categories:")
    
    print("\n1. **Resource Constraints:**")
    print("   • Budget constraint: Total cost ≤ budget")
    print("   • Time constraint: Total time ≤ deadline")
    print("   • Capacity constraint: Edge capacity limits")
    print("   • Material constraint: Limited edge types")
    
    print("\n2. **Structural Constraints:**")
    print("   • Degree constraint: Max vertex degree")
    print("   • Diameter constraint: Max path length")
    print("   • Connectivity constraint: k-connectivity")
    print("   • Planarity constraint: Planar embedding")
    
    print("\n3. **Categorical Constraints:**")
    print("   • Color constraint: Edge/vertex color limits")
    print("   • Type constraint: Edge type restrictions")
    print("   • Priority constraint: Preference ordering")
    print("   • Forbidden constraint: Excluded elements")
    
    print("\n4. **Dynamic Constraints:**")
    print("   • Time-varying costs: Temporal optimization")
    print("   • Online constraints: Incremental updates")
    print("   • Stochastic constraints: Uncertain parameters")
    print("   • Adaptive constraints: Learning-based limits")
    
    print("\n5. **Multi-Objective Constraints:**")
    print("   • Cost vs reliability trade-offs")
    print("   • Speed vs quality optimization")
    print("   • Local vs global optimality")
    print("   • Pareto-optimal solutions")

def demonstrate_approximation_algorithms():
    """Demonstrate approximation algorithms for constrained MST"""
    print("\n=== Approximation Algorithms for Constrained MST ===")
    
    print("NP-Hard Constrained MST Problems:")
    
    print("\n1. **Degree-Constrained MST:**")
    print("   • Problem: MST with max degree k")
    print("   • Complexity: NP-hard for k ≥ 2")
    print("   • Approximation: (1 + 2/k)-approximation")
    print("   • Algorithm: Modify Prim's with degree tracking")
    
    print("\n2. **Capacitated MST:**")
    print("   • Problem: MST with subtree capacity limits")
    print("   • Complexity: NP-hard")
    print("   • Approximation: 2-approximation")
    print("   • Algorithm: Cluster-based decomposition")
    
    print("\n3. **Multi-Objective MST:**")
    print("   • Problem: Optimize multiple objectives")
    print("   • Complexity: Generally NP-hard")
    print("   • Approximation: Pareto frontier approximation")
    print("   • Algorithm: Weighted sum or ε-constraint method")
    
    print("\n4. **Steiner Tree with Constraints:**")
    print("   • Problem: Connect subset with constraints")
    print("   • Complexity: NP-hard")
    print("   • Approximation: 2-approximation for metric case")
    print("   • Algorithm: MST-based approximation")
    
    print("\nApproximation Techniques:")
    print("• **Greedy modification:** Adapt standard algorithms")
    print("• **LP relaxation:** Linear programming bounds")
    print("• **Primal-dual:** Dual optimization approach")
    print("• **Local search:** Iterative improvement")
    print("• **Randomized rounding:** Probabilistic techniques")

def analyze_practical_applications():
    """Analyze practical applications of constrained MST"""
    print("\n=== Practical Applications of Constrained MST ===")
    
    print("1. **Network Design:**")
    print("   • Telecommunication networks with degree limits")
    print("   • Power grids with capacity constraints")
    print("   • Transportation networks with budget limits")
    print("   • Computer networks with reliability requirements")
    
    print("\n2. **VLSI Design:**")
    print("   • Circuit layout with area constraints")
    print("   • Wire routing with layer restrictions")
    print("   • Power distribution with thermal limits")
    print("   • Clock tree synthesis with skew constraints")
    
    print("\n3. **Logistics and Supply Chain:**")
    print("   • Distribution networks with capacity limits")
    print("   • Delivery routes with time windows")
    print("   • Warehouse connections with cost budgets")
    print("   • Multi-modal transportation optimization")
    
    print("\n4. **Social Network Analysis:**")
    print("   • Information propagation with influence limits")
    print("   • Community detection with size constraints")
    print("   • Recommendation systems with diversity requirements")
    print("   • Privacy-preserving network design")
    
    print("\n5. **Environmental Planning:**")
    print("   • Habitat connectivity with land use constraints")
    print("   • Water distribution with environmental limits")
    print("   • Renewable energy networks with sustainability goals")
    print("   • Urban planning with zoning restrictions")
    
    print("\nKey Design Considerations:")
    print("• **Trade-offs:** Performance vs constraints")
    print("• **Feasibility:** Constraint satisfiability")
    print("• **Robustness:** Solution stability")
    print("• **Scalability:** Algorithm efficiency")
    print("• **Adaptability:** Dynamic constraint handling")

if __name__ == "__main__":
    test_constrained_mst()
    demonstrate_constraint_impact()
    analyze_constraint_types()
    demonstrate_approximation_algorithms()
    analyze_practical_applications()

"""
Constrained MST Concepts:
1. Budget and Resource Constraint MST Algorithms
2. Structural Constraint MST (Degree, Distance)
3. Categorical Constraint MST (Color, Priority, Forbidden)
4. Dynamic MST with Edge Updates
5. Multi-Objective MST Optimization
6. Approximation Algorithms for NP-Hard Variants
7. Capacitated and Steiner Tree Extensions

Key Constraint Categories:
- Resource constraints: budget, capacity, material limits
- Structural constraints: degree, diameter, connectivity
- Categorical constraints: color, type, priority preferences
- Dynamic constraints: time-varying, online updates
- Multi-objective constraints: trade-off optimization

Algorithm Adaptations:
- Modify standard MST algorithms (Kruskal's, Prim's)
- Add constraint checking during edge selection
- Use approximation algorithms for NP-hard variants
- Employ heuristics for complex constraint combinations

Complexity Considerations:
- Many constrained MST problems are NP-hard
- Approximation algorithms provide practical solutions
- Greedy modifications often work well in practice
- Trade-offs between optimality and efficiency

Real-world Applications:
- Network design with capacity and degree limits
- VLSI circuit layout with area and timing constraints
- Supply chain optimization with resource limits
- Social network analysis with diversity requirements
- Environmental planning with sustainability constraints

This collection demonstrates advanced MST techniques
for real-world optimization problems with constraints.
"""
