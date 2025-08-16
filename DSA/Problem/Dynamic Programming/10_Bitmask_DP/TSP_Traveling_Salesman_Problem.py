"""
Classic Traveling Salesman Problem (TSP)
Difficulty: Hard
Category: Bitmask DP - Classic Optimization Problem

PROBLEM DESCRIPTION:
===================
Given a list of cities and the distances between each pair of cities, find the shortest possible route that visits each city exactly once and returns to the starting city.

This is the classic NP-hard optimization problem that demonstrates the power of dynamic programming with bitmasks for solving exponential problems optimally.

Input: Distance matrix where dist[i][j] represents the distance from city i to city j
Output: Minimum total distance and the optimal route

Example:
Distance Matrix:
    0  1  2  3
0 [ 0 10 15 20]
1 [10  0 35 25]
2 [15 35  0 30]
3 [20 25 30  0]

Optimal route: 0 -> 2 -> 3 -> 1 -> 0 with total distance 80

Constraints:
- 2 <= n <= 20 (practical limit due to exponential complexity)
- Distance matrix is symmetric for undirected graphs
- All distances are non-negative
"""


def tsp_brute_force(dist):
    """
    BRUTE FORCE APPROACH:
    ====================
    Try all possible permutations of cities.
    
    Time Complexity: O(n!) - factorial explosion
    Space Complexity: O(n) - recursion stack
    """
    from itertools import permutations
    
    n = len(dist)
    if n <= 1:
        return 0, []
    
    min_cost = float('inf')
    best_path = []
    
    # Try all permutations starting from city 0
    for perm in permutations(range(1, n)):
        path = [0] + list(perm) + [0]
        cost = 0
        
        for i in range(len(path) - 1):
            cost += dist[path[i]][path[i + 1]]
        
        if cost < min_cost:
            min_cost = cost
            best_path = path[:-1]  # Remove duplicate starting city
    
    return min_cost, best_path


def tsp_bitmask_dp(dist):
    """
    BITMASK DP APPROACH:
    ===================
    Use bitmask DP to solve TSP optimally.
    
    Time Complexity: O(n^2 * 2^n) - optimal for exact TSP solution
    Space Complexity: O(n * 2^n) - DP table
    """
    n = len(dist)
    if n <= 1:
        return 0, []
    
    # dp[mask][last] = minimum cost to visit all cities in mask ending at city 'last'
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[(-1, -1)] * n for _ in range(1 << n)]
    
    # Base case: start at city 0
    dp[1][0] = 0  # mask=1 means only city 0 is visited
    
    # Fill DP table
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == float('inf'):
                continue
            
            # Try going to each unvisited city
            for next_city in range(n):
                if mask & (1 << next_city):  # Already visited
                    continue
                
                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]
                
                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = (mask, last)
    
    # Find minimum cost tour (return to starting city)
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    best_last = -1
    
    for last in range(1, n):  # Don't consider starting city as last
        total_cost = dp[full_mask][last] + dist[last][0]
        if total_cost < min_cost:
            min_cost = total_cost
            best_last = last
    
    # Reconstruct path
    def reconstruct_path():
        if best_last == -1:
            return []
        
        path = []
        mask, last = full_mask, best_last
        
        while mask != 1:  # Until only starting city remains
            path.append(last)
            prev_mask, prev_last = parent[mask][last]
            mask, last = prev_mask, prev_last
        
        path.append(0)  # Add starting city
        return path[::-1]  # Reverse to get correct order
    
    return min_cost, reconstruct_path()


def tsp_optimized_dp(dist):
    """
    OPTIMIZED BITMASK DP:
    ====================
    Memory-optimized version with better constants.
    
    Time Complexity: O(n^2 * 2^n) - same asymptotic complexity
    Space Complexity: O(n * 2^n) - optimized memory usage
    """
    n = len(dist)
    if n <= 1:
        return 0, []
    
    # Use dictionary for sparse storage
    dp = {}
    parent = {}
    
    # Initialize starting state
    dp[(1, 0)] = 0  # (mask, last_city) -> min_cost
    
    # Process all possible states
    for num_visited in range(1, n):
        for mask in range(1, 1 << n):
            if bin(mask).count('1') != num_visited:
                continue
            
            for last in range(n):
                if not (mask & (1 << last)) or (mask, last) not in dp:
                    continue
                
                current_cost = dp[(mask, last)]
                
                # Extend to unvisited cities
                for next_city in range(n):
                    if mask & (1 << next_city):
                        continue
                    
                    new_mask = mask | (1 << next_city)
                    new_cost = current_cost + dist[last][next_city]
                    
                    if (new_mask, next_city) not in dp or new_cost < dp[(new_mask, next_city)]:
                        dp[(new_mask, next_city)] = new_cost
                        parent[(new_mask, next_city)] = (mask, last)
    
    # Find optimal complete tour
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    best_end = -1
    
    for end_city in range(1, n):
        if (full_mask, end_city) in dp:
            total_cost = dp[(full_mask, end_city)] + dist[end_city][0]
            if total_cost < min_cost:
                min_cost = total_cost
                best_end = end_city
    
    # Reconstruct optimal path
    def build_path():
        if best_end == -1:
            return []
        
        path = []
        state = (full_mask, best_end)
        
        while state in parent:
            path.append(state[1])
            state = parent[state]
        
        path.append(0)
        return path[::-1]
    
    return min_cost, build_path()


def tsp_with_path_analysis(dist):
    """
    TSP WITH DETAILED PATH ANALYSIS:
    ===============================
    Solve TSP and provide detailed analysis of the solution.
    
    Time Complexity: O(n^2 * 2^n) - standard TSP DP
    Space Complexity: O(n * 2^n) - DP table + analysis data
    """
    n = len(dist)
    
    # Distance matrix analysis
    analysis = {
        'num_cities': n,
        'distance_matrix': [row[:] for row in dist],
        'total_edges': n * (n - 1) // 2,
        'min_edge': float('inf'),
        'max_edge': 0,
        'avg_edge': 0,
        'path_analysis': {},
        'state_space_size': n * (2 ** n)
    }
    
    # Analyze distance matrix
    edge_sum = 0
    edge_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            edge_weight = dist[i][j]
            analysis['min_edge'] = min(analysis['min_edge'], edge_weight)
            analysis['max_edge'] = max(analysis['max_edge'], edge_weight)
            edge_sum += edge_weight
            edge_count += 1
    
    analysis['avg_edge'] = edge_sum / edge_count if edge_count > 0 else 0
    
    # Solve TSP with detailed tracking
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[(-1, -1)] * n for _ in range(1 << n)]
    
    dp[1][0] = 0
    states_computed = 0
    
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == float('inf'):
                continue
            
            states_computed += 1
            
            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                
                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]
                
                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = (mask, last)
    
    # Find optimal solution
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    best_last = -1
    
    for last in range(1, n):
        total_cost = dp[full_mask][last] + dist[last][0]
        if total_cost < min_cost:
            min_cost = total_cost
            best_last = last
    
    # Reconstruct and analyze path
    def analyze_path():
        if best_last == -1:
            return [], {}
        
        path = []
        path_edges = []
        mask, last = full_mask, best_last
        
        while mask != 1:
            path.append(last)
            prev_mask, prev_last = parent[mask][last]
            if prev_last != -1:
                path_edges.append((prev_last, last, dist[prev_last][last]))
            mask, last = prev_mask, prev_last
        
        path.append(0)
        path = path[::-1]
        
        # Add return edge
        if len(path) > 1:
            path_edges.append((path[-1], path[0], dist[path[-1]][path[0]]))
        
        path_analysis = {
            'edges': path_edges,
            'num_edges': len(path_edges),
            'edge_costs': [edge[2] for edge in path_edges],
            'min_edge_cost': min(edge[2] for edge in path_edges) if path_edges else 0,
            'max_edge_cost': max(edge[2] for edge in path_edges) if path_edges else 0,
            'avg_edge_cost': sum(edge[2] for edge in path_edges) / len(path_edges) if path_edges else 0
        }
        
        return path, path_analysis
    
    optimal_path, path_info = analyze_path()
    
    analysis['optimal_cost'] = min_cost
    analysis['optimal_path'] = optimal_path
    analysis['path_analysis'] = path_info
    analysis['states_computed'] = states_computed
    analysis['computation_efficiency'] = states_computed / analysis['state_space_size']
    
    return min_cost, optimal_path, analysis


def tsp_analysis(dist):
    """
    COMPREHENSIVE TSP ANALYSIS:
    ==========================
    Analyze the TSP instance and solution with detailed insights.
    """
    n = len(dist)
    print(f"Traveling Salesman Problem Analysis:")
    print(f"Number of cities: {n}")
    print(f"Distance matrix:")
    
    # Print distance matrix
    print("    ", end="")
    for j in range(n):
        print(f"{j:4}", end="")
    print()
    
    for i in range(n):
        print(f"{i}: ", end="")
        for j in range(n):
            print(f"{dist[i][j]:4}", end="")
        print()
    
    print(f"\nProblem complexity:")
    print(f"Possible tours: {factorial(n-1) // 2:,} (for symmetric TSP)")
    print(f"DP state space: {n * (2**n):,}")
    
    # Different approaches
    if n <= 8:
        try:
            brute_cost, brute_path = tsp_brute_force(dist)
            print(f"Brute force result: cost={brute_cost}, path={brute_path}")
        except:
            print("Brute force: Too slow")
    
    dp_cost, dp_path = tsp_bitmask_dp(dist)
    opt_cost, opt_path = tsp_optimized_dp(dist)
    
    print(f"Bitmask DP result: cost={dp_cost}, path={dp_path}")
    print(f"Optimized DP result: cost={opt_cost}, path={opt_path}")
    
    # Detailed analysis
    if n <= 12:
        detailed_cost, detailed_path, analysis = tsp_with_path_analysis(dist)
        
        print(f"\nDetailed Analysis:")
        print(f"Optimal tour cost: {detailed_cost}")
        print(f"Optimal tour path: {detailed_path}")
        print(f"States computed: {analysis['states_computed']:,}/{analysis['state_space_size']:,}")
        print(f"Computation efficiency: {analysis['computation_efficiency']:.2%}")
        
        print(f"\nDistance Matrix Statistics:")
        print(f"Minimum edge: {analysis['min_edge']}")
        print(f"Maximum edge: {analysis['max_edge']}")
        print(f"Average edge: {analysis['avg_edge']:.2f}")
        
        if analysis['path_analysis']:
            path_info = analysis['path_analysis']
            print(f"\nOptimal Path Analysis:")
            print(f"Number of edges: {path_info['num_edges']}")
            print(f"Edge costs: {path_info['edge_costs']}")
            print(f"Min edge in path: {path_info['min_edge_cost']}")
            print(f"Max edge in path: {path_info['max_edge_cost']}")
            print(f"Avg edge in path: {path_info['avg_edge_cost']:.2f}")
            
            print(f"\nPath edges:")
            for i, (from_city, to_city, cost) in enumerate(path_info['edges']):
                print(f"  {i+1}: {from_city} -> {to_city} (cost: {cost})")
    
    return dp_cost, dp_path


def factorial(n):
    """Helper function to compute factorial"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def tsp_variants():
    """
    TSP VARIANTS AND APPLICATIONS:
    ==============================
    Different TSP formulations and related problems.
    """
    
    def asymmetric_tsp(dist):
        """Solve asymmetric TSP where dist[i][j] != dist[j][i]"""
        # Same algorithm works for asymmetric case
        return tsp_bitmask_dp(dist)
    
    def tsp_with_time_windows(dist, time_windows):
        """TSP with time constraints (simplified version)"""
        # This is much more complex - simplified approximation
        cost, path = tsp_bitmask_dp(dist)
        
        # Check if path satisfies time windows (simplified check)
        current_time = 0
        valid = True
        
        for i in range(len(path) - 1):
            current_time += dist[path[i]][path[i + 1]]
            if path[i + 1] in time_windows:
                earliest, latest = time_windows[path[i + 1]]
                if current_time < earliest or current_time > latest:
                    valid = False
                    break
        
        return cost if valid else float('inf'), path if valid else []
    
    def bottleneck_tsp(dist):
        """Minimize the maximum edge in the tour"""
        n = len(dist)
        
        # Binary search on the answer
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append(dist[i][j])
        
        edges = sorted(set(edges))
        
        def can_tour_with_max_edge(max_edge):
            # Create modified distance matrix
            modified_dist = [[float('inf')] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        modified_dist[i][j] = 0
                    elif dist[i][j] <= max_edge:
                        modified_dist[i][j] = dist[i][j]
            
            cost, path = tsp_bitmask_dp(modified_dist)
            return cost != float('inf')
        
        # Binary search
        left, right = 0, len(edges) - 1
        result = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            if can_tour_with_max_edge(edges[mid]):
                result = edges[mid]
                right = mid - 1
            else:
                left = mid + 1
        
        return result
    
    def k_tsp(dist, k):
        """Visit all cities using k salesmen"""
        # This is much more complex - simplified version
        n = len(dist)
        if k >= n:
            return 0  # Each salesman visits at most one city
        
        # Approximate by partitioning cities and solving TSP for each partition
        cities_per_salesman = n // k
        total_cost = 0
        
        for i in range(k):
            start_idx = i * cities_per_salesman
            end_idx = start_idx + cities_per_salesman if i < k - 1 else n
            
            if end_idx > start_idx + 1:
                # Create subproblem
                sub_cities = list(range(start_idx, end_idx))
                sub_dist = [[dist[sub_cities[i]][sub_cities[j]] 
                           for j in range(len(sub_cities))] 
                          for i in range(len(sub_cities))]
                
                sub_cost, _ = tsp_bitmask_dp(sub_dist)
                total_cost += sub_cost
        
        return total_cost
    
    # Test variants
    test_matrices = [
        # Symmetric small instance
        [[0, 10, 15, 20],
         [10, 0, 35, 25],
         [15, 35, 0, 30],
         [20, 25, 30, 0]],
        
        # Asymmetric instance
        [[0, 10, 15, 20],
         [5, 0, 9, 10],
         [6, 13, 0, 12],
         [8, 8, 9, 0]]
    ]
    
    print("TSP Variants:")
    print("=" * 50)
    
    for i, dist in enumerate(test_matrices):
        print(f"\nDistance Matrix {i + 1}:")
        
        symmetric = all(dist[i][j] == dist[j][i] for i in range(len(dist)) for j in range(len(dist)))
        print(f"Symmetric: {symmetric}")
        
        # Basic TSP
        cost, path = tsp_bitmask_dp(dist)
        print(f"Basic TSP: cost={cost}, path={path}")
        
        # Asymmetric TSP
        asym_cost, asym_path = asymmetric_tsp(dist)
        print(f"Asymmetric TSP: cost={asym_cost}, path={asym_path}")
        
        # Bottleneck TSP
        bottleneck_result = bottleneck_tsp(dist)
        print(f"Bottleneck TSP (max edge): {bottleneck_result}")
        
        # k-TSP
        if len(dist) >= 4:
            k_result = k_tsp(dist, 2)
            print(f"2-TSP approximation: {k_result}")
        
        # Time windows example
        time_windows = {1: (10, 20), 2: (20, 30)}
        tw_cost, tw_path = tsp_with_time_windows(dist, time_windows)
        print(f"TSP with time windows: cost={tw_cost}, path={tw_path}")


# Test cases
def test_tsp():
    """Test TSP implementations with various instances"""
    test_cases = [
        # Small symmetric instance
        ([[0, 10, 15, 20],
          [10, 0, 35, 25],
          [15, 35, 0, 30],
          [20, 25, 30, 0]], 80),
        
        # Triangular instance
        ([[0, 1, 2],
          [1, 0, 1],
          [2, 1, 0]], 4),
        
        # Single city
        ([[0]], 0),
        
        # Two cities
        ([[0, 5],
          [5, 0]], 10)
    ]
    
    print("Testing Traveling Salesman Problem Solutions:")
    print("=" * 70)
    
    for i, (dist, expected) in enumerate(test_cases):
        n = len(dist)
        print(f"\nTest Case {i + 1} (n={n}):")
        print(f"Expected cost: {expected}")
        
        # Skip brute force for large instances
        if n <= 6:
            try:
                brute_cost, brute_path = tsp_brute_force(dist)
                print(f"Brute Force:      cost={brute_cost:>4}, path={brute_path} {'✓' if brute_cost == expected else '✗'}")
            except:
                print(f"Brute Force:      Timeout")
        
        if n >= 2:
            dp_cost, dp_path = tsp_bitmask_dp(dist)
            opt_cost, opt_path = tsp_optimized_dp(dist)
            
            print(f"Bitmask DP:       cost={dp_cost:>4}, path={dp_path} {'✓' if dp_cost == expected else '✗'}")
            print(f"Optimized DP:     cost={opt_cost:>4}, path={opt_path} {'✓' if opt_cost == expected else '✗'}")
    
    # Detailed analysis example
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS EXAMPLE:")
    print("-" * 40)
    example_dist = [[0, 10, 15, 20],
                    [10, 0, 35, 25],
                    [15, 35, 0, 30],
                    [20, 25, 30, 0]]
    tsp_analysis(example_dist)
    
    # Variants demonstration
    print(f"\n" + "=" * 70)
    tsp_variants()
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("1. EXPONENTIAL OPTIMIZATION: TSP is NP-hard but solvable exactly for small instances")
    print("2. BITMASK STATE COMPRESSION: Represent visited cities efficiently")
    print("3. OPTIMAL SUBSTRUCTURE: Optimal tour built from optimal sub-tours")
    print("4. HAMILTONIAN CYCLE: Visit each vertex exactly once and return to start")
    print("5. PRACTICAL APPLICATIONS: Logistics, circuit design, DNA sequencing")
    
    print("\n" + "=" * 70)
    print("Applications:")
    print("• Logistics: Vehicle routing and delivery optimization")
    print("• Manufacturing: Circuit board drilling and component placement")
    print("• Bioinformatics: DNA fragment assembly and phylogenetic analysis")
    print("• Computer Graphics: Pen plotter path optimization")
    print("• Operations Research: Classical optimization problem")


if __name__ == "__main__":
    test_tsp()


"""
TRAVELING SALESMAN PROBLEM - CLASSIC BITMASK DP OPTIMIZATION:
=============================================================

The TSP represents the pinnacle of combinatorial optimization problems:
- Visit all cities exactly once and return to start
- Minimize total travel distance/cost
- NP-hard problem with exponential solution space
- Optimal solution achievable for small instances using Bitmask DP

KEY INSIGHTS:
============
1. **EXPONENTIAL COMPLEXITY**: TSP is NP-hard with (n-1)!/2 possible tours
2. **BITMASK STATE COMPRESSION**: Represent visited cities using bit patterns
3. **OPTIMAL SUBSTRUCTURE**: Optimal tour constructed from optimal sub-tours
4. **HAMILTONIAN CYCLE**: Must visit each vertex exactly once and return to origin
5. **REAL-WORLD SIGNIFICANCE**: Foundation for numerous practical optimization problems

ALGORITHM APPROACHES:
====================

1. **Brute Force**: O(n!) time, O(n) space
   - Enumerate all possible permutations
   - Guaranteed optimal but exponentially slow

2. **Bitmask DP**: O(n² × 2^n) time, O(n × 2^n) space
   - Dynamic programming with state compression
   - Optimal solution for instances up to n≈20

3. **Optimized DP**: O(n² × 2^n) time, O(n × 2^n) space
   - Memory-efficient implementation
   - Better constant factors and cache performance

4. **Advanced Analysis**: O(n² × 2^n) time, O(n × 2^n) space
   - Includes detailed path analysis and statistics
   - Comprehensive problem characterization

CORE TSP BITMASK DP ALGORITHM:
=============================
```python
def tsp_bitmask_dp(dist):
    n = len(dist)
    
    # dp[mask][last] = min cost to visit cities in mask, ending at last
    dp = [[float('inf')] * n for _ in range(1 << n)]
    dp[1][0] = 0  # Start at city 0
    
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)) or dp[mask][last] == float('inf'):
                continue
            
            # Try extending to each unvisited city
            for next_city in range(n):
                if mask & (1 << next_city):  # Already visited
                    continue
                
                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + dist[last][next_city]
                dp[new_mask][next_city] = min(dp[new_mask][next_city], new_cost)
    
    # Find minimum cost to return to start
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    for last in range(1, n):
        min_cost = min(min_cost, dp[full_mask][last] + dist[last][0])
    
    return min_cost
```

BITMASK STATE REPRESENTATION:
============================
**City Encoding**: Each bit represents whether a city has been visited
- `mask = 0000001` → Only city 0 visited
- `mask = 0001011` → Cities 0, 1, and 3 visited
- `mask = 1111111` → All cities visited (for n=7)

**State Transitions**: `new_mask = current_mask | (1 << next_city)`
**Completion Check**: `mask == (1 << n) - 1`

DP STATE DESIGN:
===============
**State Definition**: `dp[mask][last]` = minimum cost to visit all cities in `mask`, currently at city `last`

**Base Case**: `dp[1][0] = 0` (start at city 0 with cost 0)

**Recurrence Relation**:
```
dp[new_mask][next] = min(dp[new_mask][next], 
                        dp[mask][last] + dist[last][next])
```

**Final Answer**: `min(dp[full_mask][i] + dist[i][0] for i in range(1, n))`

PATH RECONSTRUCTION:
===================
**Parent Tracking**: Store previous state and city for each optimal transition
```python
parent[new_mask][next_city] = (mask, last_city)
```

**Backtracking**: Follow parent pointers from final state to reconstruct optimal tour
```python
def reconstruct_path():
    path = []
    state = (full_mask, best_last_city)
    
    while state in parent:
        path.append(state[1])
        state = parent[state]
    
    path.append(0)  # Add starting city
    return path[::-1]  # Reverse for correct order
```

COMPLEXITY ANALYSIS:
===================
**Time Complexity**: O(n² × 2^n)
- 2^n possible subsets of cities
- n possible ending cities per subset
- n transitions per state

**Space Complexity**: O(n × 2^n)
- DP table storage
- Parent tracking for path reconstruction

**Practical Limits**: n ≤ 20-25 depending on available memory and time

OPTIMIZATION TECHNIQUES:
=======================
**Memory Optimization**: Use sparse data structures for large instances
**State Pruning**: Eliminate obviously suboptimal states early
**Bit Manipulation**: Efficient subset operations using bitwise operators
**Cache Optimization**: Process states in order that maximizes cache hits

TSP VARIANTS:
============
**Asymmetric TSP**: Distance matrix not symmetric (dist[i][j] ≠ dist[j][i])
**Bottleneck TSP**: Minimize maximum edge weight in tour
**Prize-Collecting TSP**: Optional cities with profits
**TSP with Time Windows**: Cities must be visited within time constraints
**Multiple TSP (k-TSP)**: Multiple salesmen starting from depot

REAL-WORLD APPLICATIONS:
=======================
**Logistics and Transportation**:
- Vehicle routing problems
- Delivery route optimization
- Public transportation planning

**Manufacturing**:
- Circuit board drilling optimization
- Component placement on assembly lines
- Tool path planning for CNC machines

**Bioinformatics**:
- DNA sequencing and fragment assembly
- Protein folding pathway analysis
- Phylogenetic tree construction

**Computer Science**:
- Network routing protocols
- Compiler optimization (instruction scheduling)
- Database query optimization

APPROXIMATION ALGORITHMS:
========================
For large instances where exact solutions are impractical:

**Nearest Neighbor**: O(n²) - simple greedy heuristic
**Christofides Algorithm**: 1.5-approximation for metric TSP
**Genetic Algorithms**: Evolutionary approach for large instances
**Simulated Annealing**: Probabilistic optimization method
**Lin-Kernighan Heuristic**: Local search improvement method

PROBLEM VARIATIONS:
==================
**Euclidean TSP**: Cities in 2D plane with Euclidean distances
**Manhattan TSP**: Grid-based distances
**Clustered TSP**: Cities grouped in clusters
**Dynamic TSP**: Distances change over time
**Stochastic TSP**: Probabilistic travel times

THEORETICAL SIGNIFICANCE:
========================
**Computational Complexity**: Canonical NP-hard problem
**Approximation Theory**: Benchmark for algorithm design
**Heuristic Development**: Testbed for optimization techniques
**Mathematical Modeling**: Graph theory and combinatorial optimization

The TSP embodies the fundamental trade-off between solution
quality and computational complexity, demonstrating how
Bitmask DP can provide exact solutions for moderately-sized
instances while highlighting the need for approximation
algorithms for larger real-world problems.
"""
