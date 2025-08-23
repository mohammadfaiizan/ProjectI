"""
Project Selection Problem - Multiple Approaches
Difficulty: Medium

The Project Selection Problem is a classic application of network flow algorithms.
Given a set of projects with profits/costs and dependencies between projects,
find the optimal subset of projects to maximize profit while respecting constraints.

Key Concepts:
1. Maximum Weight Closure Problem
2. Min-Cut Max-Flow Application
3. Project Dependencies and Prerequisites
4. Profit Maximization with Constraints
5. Network Flow Modeling Techniques
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from collections import defaultdict, deque
import heapq

class ProjectSelectionProblem:
    """Multiple approaches to solve project selection optimization"""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset algorithm statistics"""
        self.stats = {
            'projects_evaluated': 0,
            'dependencies_processed': 0,
            'flow_computations': 0,
            'iterations': 0
        }
    
    def max_profit_basic_flow(self, profits: List[int], dependencies: List[Tuple[int, int]]) -> Dict:
        """
        Approach 1: Basic Max-Flow Modeling
        
        Model project selection as max-flow problem with source/sink connections.
        
        Time: O(V * E^2) using Edmonds-Karp
        Space: O(V + E)
        """
        self.reset_statistics()
        n = len(profits)
        
        if n == 0:
            return {'max_profit': 0, 'selected_projects': [], 'algorithm': 'basic_flow'}
        
        # Create flow network
        # Vertices: source(0), projects(1..n), sink(n+1)
        source = 0
        sink = n + 1
        total_vertices = n + 2
        
        # Build adjacency list for flow network
        graph = defaultdict(lambda: defaultdict(int))
        
        # Add edges based on profits
        total_positive_profit = 0
        
        for i, profit in enumerate(profits):
            project_id = i + 1
            
            if profit > 0:
                # Positive profit: source to project
                graph[source][project_id] = profit
                total_positive_profit += profit
            else:
                # Negative profit: project to sink
                graph[project_id][sink] = -profit
            
            self.stats['projects_evaluated'] += 1
        
        # Add dependency edges (infinite capacity)
        for prereq, dependent in dependencies:
            prereq_id = prereq + 1
            dependent_id = dependent + 1
            graph[prereq_id][dependent_id] = float('inf')
            self.stats['dependencies_processed'] += 1
        
        # Find maximum flow (minimum cut)
        max_flow_result = self._edmonds_karp_max_flow(graph, source, sink)
        max_flow = max_flow_result['max_flow']
        
        # Maximum profit = total positive profit - max flow
        max_profit = total_positive_profit - max_flow
        
        # Find selected projects (reachable from source in residual graph)
        selected_projects = self._find_selected_projects(graph, source, sink, n)
        
        return {
            'max_profit': max_profit,
            'selected_projects': selected_projects,
            'total_positive_profit': total_positive_profit,
            'min_cut_value': max_flow,
            'algorithm': 'basic_flow',
            'statistics': self.stats.copy()
        }
    
    def max_profit_weighted_closure(self, profits: List[int], dependencies: List[Tuple[int, int]]) -> Dict:
        """
        Approach 2: Maximum Weight Closure Problem
        
        Direct application of maximum weight closure using min-cut.
        
        Time: O(V * E^2)
        Space: O(V + E)
        """
        self.reset_statistics()
        n = len(profits)
        
        if n == 0:
            return {'max_profit': 0, 'selected_projects': [], 'algorithm': 'weighted_closure'}
        
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = [0] * n
        
        for prereq, dependent in dependencies:
            graph[prereq].append(dependent)
            in_degree[dependent] += 1
            self.stats['dependencies_processed'] += 1
        
        # Create flow network for maximum weight closure
        source = n
        sink = n + 1
        flow_graph = defaultdict(lambda: defaultdict(int))
        
        total_positive_weight = 0
        
        for i in range(n):
            if profits[i] > 0:
                # Positive weight: source to vertex
                flow_graph[source][i] = profits[i]
                total_positive_weight += profits[i]
            elif profits[i] < 0:
                # Negative weight: vertex to sink
                flow_graph[i][sink] = -profits[i]
            
            self.stats['projects_evaluated'] += 1
        
        # Add dependency edges with infinite capacity
        for prereq, dependent in dependencies:
            flow_graph[prereq][dependent] = float('inf')
        
        # Find minimum cut
        max_flow_result = self._edmonds_karp_max_flow(flow_graph, source, sink)
        min_cut_value = max_flow_result['max_flow']
        
        # Maximum weight closure = total positive weight - min cut
        max_profit = total_positive_weight - min_cut_value
        
        # Find closure (projects reachable from source)
        selected_projects = self._find_reachable_projects(flow_graph, source, n)
        
        return {
            'max_profit': max_profit,
            'selected_projects': selected_projects,
            'total_positive_weight': total_positive_weight,
            'min_cut_value': min_cut_value,
            'algorithm': 'weighted_closure',
            'statistics': self.stats.copy()
        }
    
    def max_profit_dynamic_programming(self, profits: List[int], dependencies: List[Tuple[int, int]]) -> Dict:
        """
        Approach 3: Dynamic Programming on DAG
        
        Use DP on dependency DAG when no cycles exist.
        
        Time: O(V + E) for topological sort + O(V * 2^V) for DP
        Space: O(V + E)
        """
        self.reset_statistics()
        n = len(profits)
        
        if n == 0:
            return {'max_profit': 0, 'selected_projects': [], 'algorithm': 'dynamic_programming'}
        
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = [0] * n
        
        for prereq, dependent in dependencies:
            graph[prereq].append(dependent)
            in_degree[dependent] += 1
            self.stats['dependencies_processed'] += 1
        
        # Check if DAG (no cycles)
        if not self._is_dag(graph, n):
            # Fall back to flow-based approach for cyclic dependencies
            return self.max_profit_basic_flow(profits, dependencies)
        
        # Topological sort
        topo_order = self._topological_sort(graph, in_degree, n)
        
        # DP: dp[i] = maximum profit considering projects 0..i
        # For each project, decide whether to include it or not
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(mask: int) -> int:
            """DP with bitmask representing selected projects"""
            total_profit = 0
            
            # Check if selection is valid (all dependencies satisfied)
            for i in range(n):
                if mask & (1 << i):  # Project i is selected
                    # Check if all prerequisites are selected
                    for prereq, dependent in dependencies:
                        if dependent == i and not (mask & (1 << prereq)):
                            return float('-inf')  # Invalid selection
                    
                    total_profit += profits[i]
            
            return total_profit
        
        # Find optimal selection
        max_profit = 0
        best_mask = 0
        
        # Try all possible selections (exponential, but works for small n)
        for mask in range(1 << n):
            profit = dp(mask)
            if profit > max_profit:
                max_profit = profit
                best_mask = mask
            
            self.stats['projects_evaluated'] += 1
        
        # Extract selected projects
        selected_projects = []
        for i in range(n):
            if best_mask & (1 << i):
                selected_projects.append(i)
        
        return {
            'max_profit': max_profit,
            'selected_projects': selected_projects,
            'algorithm': 'dynamic_programming',
            'statistics': self.stats.copy()
        }
    
    def max_profit_greedy_heuristic(self, profits: List[int], dependencies: List[Tuple[int, int]]) -> Dict:
        """
        Approach 4: Greedy Heuristic
        
        Fast approximation using greedy selection based on profit-to-dependency ratio.
        
        Time: O(V log V + E)
        Space: O(V + E)
        """
        self.reset_statistics()
        n = len(profits)
        
        if n == 0:
            return {'max_profit': 0, 'selected_projects': [], 'algorithm': 'greedy_heuristic'}
        
        # Build dependency graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        in_degree = [0] * n
        out_degree = [0] * n
        
        for prereq, dependent in dependencies:
            graph[prereq].append(dependent)
            reverse_graph[dependent].append(prereq)
            in_degree[dependent] += 1
            out_degree[prereq] += 1
            self.stats['dependencies_processed'] += 1
        
        # Calculate priority for each project
        priorities = []
        for i in range(n):
            # Priority = profit / (1 + dependencies_required)
            dependency_cost = len(reverse_graph[i])
            if profits[i] > 0:
                priority = profits[i] / (1 + dependency_cost)
            else:
                priority = profits[i] * (1 + dependency_cost)
            
            priorities.append((priority, i))
            self.stats['projects_evaluated'] += 1
        
        # Sort by priority (descending)
        priorities.sort(reverse=True)
        
        # Greedy selection
        selected = set()
        total_profit = 0
        
        for priority, project in priorities:
            # Check if all dependencies are satisfied
            can_select = True
            for prereq in reverse_graph[project]:
                if prereq not in selected:
                    can_select = False
                    break
            
            if can_select and profits[project] > 0:
                selected.add(project)
                total_profit += profits[project]
            elif can_select and profits[project] <= 0:
                # Only select negative profit projects if they enable profitable ones
                future_profit = self._calculate_future_profit(project, graph, profits, selected)
                if future_profit + profits[project] > 0:
                    selected.add(project)
                    total_profit += profits[project]
        
        return {
            'max_profit': total_profit,
            'selected_projects': list(selected),
            'algorithm': 'greedy_heuristic',
            'statistics': self.stats.copy()
        }
    
    def max_profit_branch_and_bound(self, profits: List[int], dependencies: List[Tuple[int, int]]) -> Dict:
        """
        Approach 5: Branch and Bound
        
        Optimal solution using branch and bound with pruning.
        
        Time: O(2^V) worst case, but with pruning
        Space: O(V)
        """
        self.reset_statistics()
        n = len(profits)
        
        if n == 0:
            return {'max_profit': 0, 'selected_projects': [], 'algorithm': 'branch_and_bound'}
        
        # Build dependency graph
        prereq_map = defaultdict(list)
        for prereq, dependent in dependencies:
            prereq_map[dependent].append(prereq)
            self.stats['dependencies_processed'] += 1
        
        self.best_profit = 0
        self.best_selection = []
        
        def upper_bound(current_selection: Set[int], remaining_projects: List[int]) -> int:
            """Calculate upper bound for remaining decisions"""
            bound = sum(profits[i] for i in current_selection)
            
            # Add all positive profits from remaining projects
            for project in remaining_projects:
                if profits[project] > 0:
                    bound += profits[project]
            
            return bound
        
        def is_valid_selection(selection: Set[int]) -> bool:
            """Check if selection satisfies all dependencies"""
            for project in selection:
                for prereq in prereq_map[project]:
                    if prereq not in selection:
                        return False
            return True
        
        def branch_and_bound(current_selection: Set[int], remaining_projects: List[int], 
                           current_profit: int):
            """Branch and bound recursive function"""
            self.stats['iterations'] += 1
            
            if not remaining_projects:
                if current_profit > self.best_profit:
                    self.best_profit = current_profit
                    self.best_selection = list(current_selection)
                return
            
            # Pruning: check upper bound
            if upper_bound(current_selection, remaining_projects) <= self.best_profit:
                return
            
            project = remaining_projects[0]
            remaining = remaining_projects[1:]
            
            # Branch 1: Don't select current project
            branch_and_bound(current_selection, remaining, current_profit)
            
            # Branch 2: Select current project (if dependencies satisfied)
            new_selection = current_selection | {project}
            if is_valid_selection(new_selection):
                new_profit = current_profit + profits[project]
                branch_and_bound(new_selection, remaining, new_profit)
        
        # Start branch and bound
        all_projects = list(range(n))
        branch_and_bound(set(), all_projects, 0)
        
        return {
            'max_profit': self.best_profit,
            'selected_projects': self.best_selection,
            'algorithm': 'branch_and_bound',
            'statistics': self.stats.copy()
        }
    
    def _edmonds_karp_max_flow(self, graph: Dict, source: int, sink: int) -> Dict:
        """Edmonds-Karp maximum flow algorithm"""
        max_flow = 0
        iterations = 0
        
        # Create residual graph
        residual = defaultdict(lambda: defaultdict(int))
        for u in graph:
            for v in graph[u]:
                residual[u][v] = graph[u][v]
        
        while True:
            iterations += 1
            self.stats['iterations'] += 1
            
            # BFS to find augmenting path
            parent = {}
            queue = deque([source])
            parent[source] = None
            
            while queue and sink not in parent:
                current = queue.popleft()
                
                for neighbor in residual[current]:
                    if neighbor not in parent and residual[current][neighbor] > 0:
                        parent[neighbor] = current
                        queue.append(neighbor)
            
            if sink not in parent:
                break
            
            # Find bottleneck capacity
            path_flow = float('inf')
            current = sink
            
            while current != source:
                prev = parent[current]
                path_flow = min(path_flow, residual[prev][current])
                current = prev
            
            # Update residual graph
            current = sink
            while current != source:
                prev = parent[current]
                residual[prev][current] -= path_flow
                residual[current][prev] += path_flow
                current = prev
            
            max_flow += path_flow
        
        self.stats['flow_computations'] += 1
        
        return {
            'max_flow': max_flow,
            'iterations': iterations,
            'residual_graph': residual
        }
    
    def _find_selected_projects(self, graph: Dict, source: int, sink: int, n: int) -> List[int]:
        """Find selected projects from min-cut"""
        # Run max-flow to get residual graph
        flow_result = self._edmonds_karp_max_flow(graph, source, sink)
        residual = flow_result['residual_graph']
        
        # Find reachable vertices from source in residual graph
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        
        while queue:
            current = queue.popleft()
            for neighbor in residual[current]:
                if neighbor not in reachable and residual[current][neighbor] > 0:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        # Extract project IDs (subtract 1 to convert back to 0-indexed)
        selected_projects = []
        for vertex in reachable:
            if 1 <= vertex <= n:  # Project vertices
                selected_projects.append(vertex - 1)
        
        return selected_projects
    
    def _find_reachable_projects(self, graph: Dict, source: int, n: int) -> List[int]:
        """Find projects reachable from source"""
        reachable = set()
        queue = deque([source])
        reachable.add(source)
        
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in reachable and graph[current][neighbor] > 0:
                    reachable.add(neighbor)
                    queue.append(neighbor)
        
        # Extract project IDs
        selected_projects = []
        for vertex in reachable:
            if 0 <= vertex < n:  # Project vertices
                selected_projects.append(vertex)
        
        return selected_projects
    
    def _is_dag(self, graph: Dict, n: int) -> bool:
        """Check if graph is a DAG (no cycles)"""
        color = [0] * n  # 0: white, 1: gray, 2: black
        
        def has_cycle(v):
            if color[v] == 1:  # Gray (currently visiting)
                return True
            if color[v] == 2:  # Black (already processed)
                return False
            
            color[v] = 1  # Mark as gray
            
            for neighbor in graph[v]:
                if has_cycle(neighbor):
                    return True
            
            color[v] = 2  # Mark as black
            return False
        
        for i in range(n):
            if color[i] == 0 and has_cycle(i):
                return False
        
        return True
    
    def _topological_sort(self, graph: Dict, in_degree: List[int], n: int) -> List[int]:
        """Topological sort using Kahn's algorithm"""
        queue = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)
        
        topo_order = []
        
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return topo_order
    
    def _calculate_future_profit(self, project: int, graph: Dict, profits: List[int], 
                                selected: Set[int]) -> int:
        """Calculate potential future profit from selecting a project"""
        future_profit = 0
        visited = set()
        
        def dfs(p):
            if p in visited:
                return 0
            visited.add(p)
            
            profit = 0
            for dependent in graph[p]:
                if dependent not in selected:
                    profit += max(0, profits[dependent]) + dfs(dependent)
            
            return profit
        
        return dfs(project)

def test_project_selection():
    """Test project selection algorithms"""
    print("=== Testing Project Selection Problem ===")
    
    # Test cases: (profits, dependencies, description)
    test_cases = [
        # Simple case
        ([10, -5, 15, -3], [(0, 1), (1, 2)], "Linear dependency chain"),
        
        # No dependencies
        ([5, -2, 8, -1, 3], [], "Independent projects"),
        
        # Complex dependencies
        ([20, -10, 15, -5, 25, -8], [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5)], "Tree dependencies"),
        
        # Cyclic dependencies (for flow-based methods)
        ([10, -3, 8], [(0, 1), (1, 2), (2, 0)], "Cyclic dependencies"),
        
        # All negative profits
        ([-1, -2, -3], [(0, 1)], "All negative profits"),
    ]
    
    solver = ProjectSelectionProblem()
    
    algorithms = [
        ("Basic Flow", solver.max_profit_basic_flow),
        ("Weighted Closure", solver.max_profit_weighted_closure),
        ("Dynamic Programming", solver.max_profit_dynamic_programming),
        ("Greedy Heuristic", solver.max_profit_greedy_heuristic),
        ("Branch & Bound", solver.max_profit_branch_and_bound),
    ]
    
    for profits, dependencies, description in test_cases:
        print(f"\n--- {description} ---")
        print(f"Profits: {profits}")
        print(f"Dependencies: {dependencies}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(profits, dependencies)
                max_profit = result['max_profit']
                selected = result['selected_projects']
                iterations = result['statistics'].get('iterations', 0)
                
                print(f"{alg_name:18} | Profit: {max_profit:3} | Selected: {selected} | Iter: {iterations:2}")
                
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

def demonstrate_project_selection_modeling():
    """Demonstrate project selection problem modeling"""
    print("\n=== Project Selection Modeling Demo ===")
    
    profits = [20, -10, 15, -5, 25]
    dependencies = [(0, 1), (1, 2), (0, 3), (3, 4)]
    
    print(f"Example Problem:")
    print(f"Projects: 0, 1, 2, 3, 4")
    print(f"Profits: {profits}")
    print(f"Dependencies: {dependencies}")
    
    print(f"\nDependency Analysis:")
    print(f"• Project 1 requires Project 0")
    print(f"• Project 2 requires Project 1 (and transitively Project 0)")
    print(f"• Project 3 requires Project 0")
    print(f"• Project 4 requires Project 3 (and transitively Project 0)")
    
    print(f"\nFlow Network Modeling:")
    print(f"• Source connects to profitable projects (0: 20, 2: 15, 4: 25)")
    print(f"• Unprofitable projects connect to sink (1: 10, 3: 5)")
    print(f"• Dependencies become infinite capacity edges")
    print(f"• Min-cut determines optimal selection")
    
    solver = ProjectSelectionProblem()
    result = solver.max_profit_basic_flow(profits, dependencies)
    
    print(f"\nOptimal Solution:")
    print(f"• Maximum profit: {result['max_profit']}")
    print(f"• Selected projects: {result['selected_projects']}")
    print(f"• Min-cut value: {result['min_cut_value']}")

def analyze_project_selection_theory():
    """Analyze theoretical aspects of project selection"""
    print("\n=== Project Selection Theory ===")
    
    print("Problem Formulation:")
    
    print("\n1. **Maximum Weight Closure:**")
    print("   • Given directed graph with vertex weights")
    print("   • Find subset S such that if v ∈ S, all successors of v are in S")
    print("   • Maximize sum of weights in S")
    print("   • Project selection is special case with dependency constraints")
    
    print("\n2. **Min-Cut Max-Flow Reduction:**")
    print("   • Create source and sink vertices")
    print("   • Positive weights: source to vertex edges")
    print("   • Negative weights: vertex to sink edges")
    print("   • Dependencies: infinite capacity edges")
    print("   • Min-cut gives optimal closure")
    
    print("\n3. **Complexity Analysis:**")
    print("   • Flow-based: O(V * E²) using Edmonds-Karp")
    print("   • DP on DAG: O(V + E) + O(2^V) for subset enumeration")
    print("   • Branch and bound: O(2^V) worst case with pruning")
    print("   • Greedy: O(V log V + E) but approximate")
    
    print("\n4. **Optimality Conditions:**")
    print("   • Flow-based methods guarantee optimal solution")
    print("   • DP exact for acyclic dependencies")
    print("   • Greedy provides approximation")
    print("   • Branch and bound optimal with complete search")
    
    print("\n5. **Practical Considerations:**")
    print("   • Dependency cycles require flow-based methods")
    print("   • Large problem instances need approximation")
    print("   • Real-world constraints may require extensions")
    print("   • Sensitivity analysis important for decision making")

if __name__ == "__main__":
    test_project_selection()
    demonstrate_project_selection_modeling()
    analyze_project_selection_theory()

"""
Project Selection Problem - Key Insights:

1. **Problem Structure:**
   - Projects with profits (positive) or costs (negative)
   - Dependencies between projects (prerequisites)
   - Goal: maximize total profit while satisfying constraints
   - Classic application of maximum weight closure

2. **Algorithm Categories:**
   - Flow-based: Min-cut max-flow for optimal solution
   - Dynamic Programming: Exact solution for DAGs
   - Greedy: Fast approximation algorithms
   - Branch and Bound: Optimal with pruning

3. **Network Flow Modeling:**
   - Source connects to profitable projects
   - Unprofitable projects connect to sink
   - Dependencies become infinite capacity edges
   - Min-cut determines optimal project selection

4. **Complexity Considerations:**
   - Flow methods: Polynomial time, optimal
   - DP methods: Exponential space, exact for DAGs
   - Greedy methods: Fast but approximate
   - Branch and bound: Exponential worst case

5. **Real-World Applications:**
   - Software project portfolio optimization
   - Investment decision making
   - Resource allocation with constraints
   - Manufacturing process selection

The project selection problem demonstrates the power
of network flow algorithms for constrained optimization.
"""
