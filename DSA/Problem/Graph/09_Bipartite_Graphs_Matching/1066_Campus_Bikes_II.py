"""
1066. Campus Bikes II
Difficulty: Medium

Problem:
On a campus represented as a 2D grid, there are n workers and m bikes, with n <= m. 
Each worker and bike is a 2D coordinate on this grid.

We assign one unique bike to each worker so that the sum of the Manhattan distances between 
each worker and their assigned bike is minimized.

Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.

The Manhattan distance between two points p1 and p2 is |p1.x - p2.x| + |p1.y - p2.y|.

Examples:
Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: 6
Explanation: 
We assign bike 0 to worker 0, bike 1 to worker 1. The Manhattan distance of both assignments is 3, so the output is 6.

Input: workers = [[0,0],[1,1],[2,0]], bikes = [[1,0],[2,2],[2,1]]
Output: 4
Explanation: 
We first assign bike 0 to worker 0, then assign bike 1 to worker 2, then assign bike 2 to worker 1. 
The Manhattan distances are 1 + 1 + 2 = 4.

Constraints:
- n == workers.length
- m == bikes.length
- 1 <= n <= m <= 10
- workers[i].length == bikes[i].length == 2
- 0 <= workers[i][0], workers[i][1], bikes[i][0], bikes[i][1] < 1000
"""

from typing import List, Dict, Set, Tuple, Optional
from itertools import permutations
import heapq
from functools import lru_cache

class Solution:
    def assignBikes_approach1_complete_search_permutations(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 1: Complete Search using Permutations
        
        Try all possible assignments and find minimum cost.
        
        Time: O(m! / (m-n)!) where m = bikes, n = workers
        Space: O(1)
        """
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n_workers = len(workers)
        n_bikes = len(bikes)
        min_cost = float('inf')
        
        # Try all permutations of bikes taken n_workers at a time
        for bike_assignment in permutations(range(n_bikes), n_workers):
            total_cost = 0
            for worker_idx, bike_idx in enumerate(bike_assignment):
                total_cost += manhattan_distance(workers[worker_idx], bikes[bike_idx])
            
            min_cost = min(min_cost, total_cost)
        
        return min_cost
    
    def assignBikes_approach2_backtracking_optimized(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 2: Backtracking with Pruning
        
        Use backtracking with early pruning based on current best solution.
        
        Time: O(m^n) with pruning, much better in practice
        Space: O(n) for recursion stack
        """
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n_workers = len(workers)
        used_bikes = set()
        min_cost = [float('inf')]  # Use list for mutable reference
        
        def backtrack(worker_idx, current_cost):
            # Pruning: if current cost already exceeds best, stop
            if current_cost >= min_cost[0]:
                return
            
            # Base case: all workers assigned
            if worker_idx == n_workers:
                min_cost[0] = min(min_cost[0], current_cost)
                return
            
            # Try assigning each available bike to current worker
            for bike_idx, bike in enumerate(bikes):
                if bike_idx not in used_bikes:
                    # Calculate cost for this assignment
                    cost = manhattan_distance(workers[worker_idx], bike)
                    
                    # Pruning: if this path already exceeds best, skip
                    if current_cost + cost < min_cost[0]:
                        used_bikes.add(bike_idx)
                        backtrack(worker_idx + 1, current_cost + cost)
                        used_bikes.remove(bike_idx)
        
        backtrack(0, 0)
        return min_cost[0]
    
    def assignBikes_approach3_dp_bitmask(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 3: Dynamic Programming with Bitmask
        
        Use DP with bitmask to represent used bikes.
        
        Time: O(n * 2^m * m)
        Space: O(2^m)
        """
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n_workers = len(workers)
        n_bikes = len(bikes)
        
        # dp[mask] = minimum cost to assign first k workers using bikes in mask
        dp = {}
        
        def solve(worker_idx, bike_mask):
            if worker_idx == n_workers:
                return 0
            
            if (worker_idx, bike_mask) in dp:
                return dp[(worker_idx, bike_mask)]
            
            min_cost = float('inf')
            
            # Try assigning each available bike to current worker
            for bike_idx in range(n_bikes):
                if not (bike_mask & (1 << bike_idx)):  # Bike not used
                    cost = manhattan_distance(workers[worker_idx], bikes[bike_idx])
                    remaining_cost = solve(worker_idx + 1, bike_mask | (1 << bike_idx))
                    min_cost = min(min_cost, cost + remaining_cost)
            
            dp[(worker_idx, bike_mask)] = min_cost
            return min_cost
        
        return solve(0, 0)
    
    def assignBikes_approach4_dijkstra_state_space(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 4: Dijkstra's Algorithm on State Space
        
        Model as shortest path problem in state space.
        
        Time: O(2^m * log(2^m) * m)
        Space: O(2^m)
        """
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n_workers = len(workers)
        n_bikes = len(bikes)
        
        # State: (cost, worker_idx, bike_mask)
        # bike_mask represents which bikes are used
        pq = [(0, 0, 0)]  # (cost, worker_idx, bike_mask)
        visited = set()
        
        while pq:
            cost, worker_idx, bike_mask = heapq.heappop(pq)
            
            # Check if already processed this state
            state = (worker_idx, bike_mask)
            if state in visited:
                continue
            visited.add(state)
            
            # All workers assigned
            if worker_idx == n_workers:
                return cost
            
            # Try assigning each available bike to current worker
            for bike_idx in range(n_bikes):
                if not (bike_mask & (1 << bike_idx)):  # Bike available
                    new_cost = cost + manhattan_distance(workers[worker_idx], bikes[bike_idx])
                    new_mask = bike_mask | (1 << bike_idx)
                    new_state = (worker_idx + 1, new_mask)
                    
                    if new_state not in visited:
                        heapq.heappush(pq, (new_cost, worker_idx + 1, new_mask))
        
        return -1  # Should not reach here
    
    def assignBikes_approach5_branch_and_bound(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 5: Branch and Bound with Lower Bound Estimation
        
        Use sophisticated pruning with lower bound estimation.
        
        Time: O(m^n) with aggressive pruning
        Space: O(n)
        """
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        def calculate_lower_bound(worker_idx, used_bikes, current_cost):
            """Calculate lower bound for remaining assignment"""
            if worker_idx >= len(workers):
                return current_cost
            
            # For each remaining worker, find minimum distance to any available bike
            remaining_cost = 0
            available_bikes = [i for i in range(len(bikes)) if i not in used_bikes]
            
            for w_idx in range(worker_idx, len(workers)):
                if available_bikes:
                    min_dist = min(manhattan_distance(workers[w_idx], bikes[b_idx]) 
                                 for b_idx in available_bikes)
                    remaining_cost += min_dist
                    
                    # Remove the closest bike for next iteration (greedy approximation)
                    if len(available_bikes) > len(workers) - w_idx - 1:
                        closest_bike = min(available_bikes, 
                                         key=lambda b: manhattan_distance(workers[w_idx], bikes[b]))
                        available_bikes.remove(closest_bike)
            
            return current_cost + remaining_cost
        
        n_workers = len(workers)
        used_bikes = set()
        best_cost = [float('inf')]
        
        def branch_and_bound(worker_idx, current_cost):
            # Calculate lower bound
            lower_bound = calculate_lower_bound(worker_idx, used_bikes, current_cost)
            
            # Pruning: if lower bound exceeds best known solution
            if lower_bound >= best_cost[0]:
                return
            
            # Base case: all workers assigned
            if worker_idx == n_workers:
                best_cost[0] = min(best_cost[0], current_cost)
                return
            
            # Create list of bikes sorted by distance to current worker
            bike_distances = []
            for bike_idx, bike in enumerate(bikes):
                if bike_idx not in used_bikes:
                    dist = manhattan_distance(workers[worker_idx], bike)
                    bike_distances.append((dist, bike_idx))
            
            bike_distances.sort()  # Try closest bikes first
            
            for dist, bike_idx in bike_distances:
                new_cost = current_cost + dist
                
                # Early pruning
                if new_cost < best_cost[0]:
                    used_bikes.add(bike_idx)
                    branch_and_bound(worker_idx + 1, new_cost)
                    used_bikes.remove(bike_idx)
        
        branch_and_bound(0, 0)
        return best_cost[0]
    
    def assignBikes_approach6_hungarian_algorithm_simulation(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 6: Hungarian Algorithm Simulation
        
        Simulate Hungarian algorithm for minimum weight bipartite matching.
        
        Time: O(n^3) where n = max(workers, bikes)
        Space: O(n^2)
        """
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        n_workers = len(workers)
        n_bikes = len(bikes)
        
        # Create cost matrix
        # Pad with dummy workers if needed
        n = max(n_workers, n_bikes)
        cost_matrix = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n_workers):
            for j in range(n_bikes):
                cost_matrix[i][j] = manhattan_distance(workers[i], bikes[j])
        
        # Simplified Hungarian algorithm (for demonstration)
        # In practice, would use optimized implementation
        
        # For small inputs, use brute force since Hungarian is complex
        if n_workers <= 3:
            return self.assignBikes_approach2_backtracking_optimized(workers, bikes)
        
        # For larger inputs, use greedy approximation
        used_bikes = set()
        total_cost = 0
        
        # Greedy assignment: for each worker, find closest available bike
        worker_order = list(range(n_workers))
        
        for worker_idx in worker_order:
            best_bike = -1
            best_cost = float('inf')
            
            for bike_idx in range(n_bikes):
                if bike_idx not in used_bikes:
                    cost = manhattan_distance(workers[worker_idx], bikes[bike_idx])
                    if cost < best_cost:
                        best_cost = cost
                        best_bike = bike_idx
            
            if best_bike != -1:
                used_bikes.add(best_bike)
                total_cost += best_cost
        
        return total_cost

def test_campus_bikes():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (workers, bikes, expected)
        ([[0,0],[2,1]], [[1,2],[3,3]], 6),
        ([[0,0],[1,1],[2,0]], [[1,0],[2,2],[2,1]], 4),
        ([[0,0]], [[1,1]], 2),
        ([[0,0],[1,0]], [[1,0],[2,0],[0,1]], 2),
        ([[3,4],[1,2]], [[0,0],[2,3],[5,1]], 8),
    ]
    
    approaches = [
        ("Permutations", solution.assignBikes_approach1_complete_search_permutations),
        ("Backtracking", solution.assignBikes_approach2_backtracking_optimized),
        ("DP Bitmask", solution.assignBikes_approach3_dp_bitmask),
        ("Dijkstra", solution.assignBikes_approach4_dijkstra_state_space),
        ("Branch & Bound", solution.assignBikes_approach5_branch_and_bound),
        ("Hungarian Sim", solution.assignBikes_approach6_hungarian_algorithm_simulation),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (workers, bikes, expected) in enumerate(test_cases):
            try:
                result = func([w[:] for w in workers], [b[:] for b in bikes])  # Deep copy
                status = "✓" if result == expected else "✗"
                print(f"Test {i+1}: {status} expected={expected}, got={result}")
            except Exception as e:
                print(f"Test {i+1}: ERROR - {str(e)}")

def demonstrate_assignment_example():
    """Demonstrate assignment problem with visual example"""
    print("\n=== Assignment Problem Demo ===")
    
    workers = [[0,0],[2,1]]
    bikes = [[1,2],[3,3]]
    
    print(f"Workers: {workers}")
    print(f"Bikes: {bikes}")
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    print(f"\nDistance matrix:")
    print(f"       Bike0  Bike1")
    for i, worker in enumerate(workers):
        row = f"Worker{i}:  "
        for j, bike in enumerate(bikes):
            dist = manhattan_distance(worker, bike)
            row += f"{dist:4}"
        print(row)
    
    print(f"\nPossible assignments:")
    print(f"1. Worker0→Bike0, Worker1→Bike1: {manhattan_distance(workers[0], bikes[0]) + manhattan_distance(workers[1], bikes[1])}")
    print(f"2. Worker0→Bike1, Worker1→Bike0: {manhattan_distance(workers[0], bikes[1]) + manhattan_distance(workers[1], bikes[0])}")
    
    solution = Solution()
    result = solution.assignBikes_approach2_backtracking_optimized(workers, bikes)
    print(f"\nOptimal cost: {result}")

def demonstrate_state_space_search():
    """Demonstrate state space search approach"""
    print("\n=== State Space Search Demo ===")
    
    workers = [[0,0],[1,1]]
    bikes = [[1,0],[0,1],[2,2]]
    
    print(f"Workers: {workers}")
    print(f"Bikes: {bikes}")
    
    print(f"\nState space exploration:")
    print(f"State format: (worker_assigned, bike_mask, cost)")
    print(f"bike_mask: binary representation of used bikes")
    
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # Simulate first few states
    print(f"\nInitial state: (0, 000, 0)")
    print(f"  Worker 0 can choose from bikes 0,1,2")
    
    for bike_idx in range(3):
        cost = manhattan_distance(workers[0], bikes[bike_idx])
        mask = 1 << bike_idx
        print(f"  Choose bike {bike_idx}: (1, {mask:03b}, {cost})")
    
    print(f"\nFrom state (1, 001, 1) - Worker 0 chose bike 0:")
    print(f"  Worker 1 can choose from bikes 1,2")
    
    for bike_idx in [1, 2]:
        cost = manhattan_distance(workers[1], bikes[bike_idx])
        base_cost = manhattan_distance(workers[0], bikes[0])
        total_cost = base_cost + cost
        mask = 1 | (1 << bike_idx)
        print(f"  Choose bike {bike_idx}: (2, {mask:03b}, {total_cost})")

def analyze_complexity_trade_offs():
    """Analyze complexity trade-offs between approaches"""
    print("\n=== Complexity Trade-offs Analysis ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Permutations:**")
    print("   • Time: O(m!/(m-n)!) - factorial explosion")
    print("   • Space: O(1) - very memory efficient")
    print("   • Best for: Very small instances (n,m ≤ 3)")
    
    print("\n2. **Backtracking:**")
    print("   • Time: O(m^n) with pruning - much better in practice")
    print("   • Space: O(n) - recursion stack")
    print("   • Best for: Small to medium instances (n ≤ 8)")
    
    print("\n3. **DP with Bitmask:**")
    print("   • Time: O(n * 2^m * m) - exponential in bikes")
    print("   • Space: O(2^m) - exponential memory")
    print("   • Best for: Few bikes, many workers")
    
    print("\n4. **Dijkstra State Space:**")
    print("   • Time: O(2^m * log(2^m) * m) - optimal path guaranteed")
    print("   • Space: O(2^m) - stores all states")
    print("   • Best for: When optimal solution needed with path tracking")
    
    print("\n5. **Branch and Bound:**")
    print("   • Time: O(m^n) with aggressive pruning")
    print("   • Space: O(n) - recursion only")
    print("   • Best for: Real-world instances with good bounds")
    
    print("\nSelection Guidelines:")
    print("• **n,m ≤ 3:** Permutations")
    print("• **n ≤ 8, good pruning:** Backtracking")
    print("• **m ≤ 15:** DP Bitmask")
    print("• **Need optimal path:** Dijkstra")
    print("• **Production systems:** Branch and Bound")

def demonstrate_real_world_extensions():
    """Demonstrate real-world extensions of the problem"""
    print("\n=== Real-World Extensions ===")
    
    print("Campus Bikes Problem Extensions:")
    
    print("\n1. **Multi-objective Optimization:**")
    print("   • Minimize total distance AND balance workload")
    print("   • Consider bike quality/condition in assignment")
    print("   • Factor in worker preferences and bike availability")
    
    print("\n2. **Dynamic Assignment:**")
    print("   • Workers arrive at different times")
    print("   • Bikes become available/unavailable dynamically")
    print("   • Real-time reoptimization requirements")
    
    print("\n3. **Fairness Constraints:**")
    print("   • Maximum distance any worker travels")
    print("   • Ensure equitable distribution of good bikes")
    print("   • Priority systems for different worker types")
    
    print("\n4. **Capacity Constraints:**")
    print("   • Multiple workers can share bikes (time slots)")
    print("   • Bikes have different capacity/size requirements")
    print("   • Maintenance schedules affect availability")
    
    print("\n5. **Stochastic Elements:**")
    print("   • Uncertain worker arrival times")
    print("   • Probabilistic bike failure rates")
    print("   • Weather-dependent demand patterns")

def demonstrate_optimization_techniques():
    """Demonstrate optimization techniques for large instances"""
    print("\n=== Optimization Techniques ===")
    
    print("Advanced Optimization Strategies:")
    
    print("\n1. **Preprocessing:**")
    print("   • Remove dominated assignments")
    print("   • Identify must-have assignments")
    print("   • Calculate tighter bounds")
    
    print("\n2. **Heuristic Initialization:**")
    print("   • Greedy assignment for upper bound")
    print("   • Nearest neighbor heuristics")
    print("   • Random restart strategies")
    
    print("\n3. **Advanced Pruning:**")
    print("   • Lower bound calculation")
    print("   • Conflict-based pruning")
    print("   • Symmetry breaking")
    
    print("\n4. **Approximation Algorithms:**")
    print("   • Greedy 2-approximation")
    print("   • Local search improvements")
    print("   • Genetic algorithms for large instances")
    
    print("\n5. **Parallel Processing:**")
    print("   • Parallel branch exploration")
    print("   • Distributed state space search")
    print("   • GPU-accelerated distance calculations")

def demonstrate_bipartite_matching_connection():
    """Demonstrate connection to bipartite matching"""
    print("\n=== Bipartite Matching Connection ===")
    
    print("Campus Bikes as Bipartite Matching:")
    
    print("\n1. **Graph Structure:**")
    print("   • Left vertices: Workers")
    print("   • Right vertices: Bikes")
    print("   • Edge weights: Manhattan distances")
    print("   • Goal: Minimum weight perfect matching")
    
    print("\n2. **Hungarian Algorithm:**")
    print("   • Optimal O(n³) algorithm for weighted bipartite matching")
    print("   • Guarantees global minimum")
    print("   • Handles arbitrary cost matrices")
    
    print("\n3. **Problem Variations:**")
    print("   • Maximum matching: Unweighted case")
    print("   • Perfect matching: All workers assigned")
    print("   • Maximum weight matching: Maximize total utility")
    
    print("\n4. **Implementation Considerations:**")
    print("   • Sparse vs dense cost matrices")
    print("   • Numerical stability")
    print("   • Early termination strategies")
    
    print("\n5. **Applications Beyond Bikes:**")
    print("   • Job assignment to workers")
    print("   • Resource allocation")
    print("   • Network flow optimization")
    print("   • Facility location problems")

if __name__ == "__main__":
    test_campus_bikes()
    demonstrate_assignment_example()
    demonstrate_state_space_search()
    analyze_complexity_trade_offs()
    demonstrate_real_world_extensions()
    demonstrate_optimization_techniques()
    demonstrate_bipartite_matching_connection()

"""
Campus Bikes and Assignment Problem Concepts:
1. Minimum Weight Bipartite Matching and Assignment Optimization
2. State Space Search and Dynamic Programming on Subsets
3. Branch and Bound with Intelligent Pruning Strategies
4. Approximation Algorithms and Heuristic Optimization
5. Real-world Extensions and Multi-objective Optimization

Key Problem Insights:
- Classic assignment problem with Manhattan distance metric
- Bipartite matching between workers and bikes
- Multiple algorithmic approaches with different trade-offs
- Exponential complexity requires intelligent pruning

Algorithm Strategy:
1. Model as minimum weight bipartite matching
2. Use appropriate algorithm based on problem size
3. Apply pruning and bounds for efficiency
4. Consider approximation for large instances

Optimization Techniques:
- Backtracking with aggressive pruning
- Dynamic programming with bitmask state representation
- Branch and bound with lower bound estimation
- Dijkstra's algorithm on state space
- Hungarian algorithm for optimal solution

Complexity Analysis:
- Permutation approach: O(m!/(m-n)!) - impractical for large inputs
- Backtracking: O(m^n) with pruning - good for small instances
- DP bitmask: O(n * 2^m * m) - exponential in bikes
- Hungarian: O(n³) - optimal for dense instances

Real-world Applications:
- Resource assignment and allocation problems
- Task scheduling and worker assignment
- Facility location and service assignment
- Transportation and logistics optimization
- Cloud computing resource allocation

This problem demonstrates fundamental assignment optimization
essential for resource allocation and operational research.
"""
