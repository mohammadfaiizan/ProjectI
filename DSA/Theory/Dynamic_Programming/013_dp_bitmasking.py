"""
Dynamic Programming - Bitmasking Patterns
This module implements various DP problems using bitmasking including TSP,
assignment problems, grid covering, and set partitioning optimization.
"""

from typing import List, Dict, Tuple, Optional, Set
import time
from functools import lru_cache

# ==================== TRAVELLING SALESMAN PROBLEM ====================

class TravellingSalesmanProblem:
    """
    Travelling Salesman Problem (TSP)
    
    Find the shortest possible route that visits each city exactly once
    and returns to the starting point using bitmask DP.
    """
    
    def tsp_min_cost(self, distances: List[List[int]]) -> int:
        """
        Find minimum cost to visit all cities and return to start
        
        Time Complexity: O(n² * 2^n)
        Space Complexity: O(n * 2^n)
        
        Args:
            distances: 2D array of distances between cities
        
        Returns:
            Minimum cost for TSP tour
        """
        n = len(distances)
        if n <= 1:
            return 0
        
        # dp[mask][i] = minimum cost to visit cities in mask ending at city i
        dp = [[float('inf')] * n for _ in range(1 << n)]
        
        # Start from city 0
        dp[1][0] = 0
        
        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    
                    new_mask = mask | (1 << v)
                    dp[new_mask][v] = min(dp[new_mask][v], 
                                         dp[mask][u] + distances[u][v])
        
        # Return to starting city
        result = float('inf')
        full_mask = (1 << n) - 1
        
        for i in range(1, n):
            result = min(result, dp[full_mask][i] + distances[i][0])
        
        return result if result != float('inf') else -1
    
    def tsp_with_path(self, distances: List[List[int]]) -> Tuple[int, List[int]]:
        """
        Find minimum cost TSP tour and the actual path
        
        Args:
            distances: 2D array of distances between cities
        
        Returns:
            Tuple of (minimum cost, path taken)
        """
        n = len(distances)
        if n <= 1:
            return 0, [0] if n == 1 else []
        
        # dp[mask][i] = minimum cost to visit cities in mask ending at city i
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[(-1, -1)] * n for _ in range(1 << n)]
        
        dp[1][0] = 0
        
        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)) or dp[mask][u] == float('inf'):
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    
                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u] + distances[u][v]
                    
                    if new_cost < dp[new_mask][v]:
                        dp[new_mask][v] = new_cost
                        parent[new_mask][v] = (mask, u)
        
        # Find best ending city and cost
        full_mask = (1 << n) - 1
        min_cost = float('inf')
        last_city = -1
        
        for i in range(1, n):
            cost = dp[full_mask][i] + distances[i][0]
            if cost < min_cost:
                min_cost = cost
                last_city = i
        
        # Reconstruct path
        path = []
        mask = full_mask
        city = last_city
        
        while city != -1:
            path.append(city)
            if parent[mask][city][0] == -1:
                break
            prev_mask, prev_city = parent[mask][city]
            mask, city = prev_mask, prev_city
        
        path.reverse()
        path.append(0)  # Return to start
        
        return min_cost, path
    
    def tsp_with_time_windows(self, distances: List[List[int]], 
                            time_windows: List[Tuple[int, int]]) -> int:
        """
        TSP with time window constraints
        
        Args:
            distances: Distance matrix
            time_windows: List of (earliest, latest) arrival times for each city
        
        Returns:
            Minimum cost TSP tour respecting time windows, -1 if impossible
        """
        n = len(distances)
        if n != len(time_windows):
            return -1
        
        dp = {}
        
        def solve(mask: int, current_city: int, current_time: int) -> int:
            if mask == (1 << n) - 1:
                # Check if we can return to start
                return_time = current_time + distances[current_city][0]
                earliest, latest = time_windows[0]
                if earliest <= return_time <= latest:
                    return distances[current_city][0]
                return float('inf')
            
            if (mask, current_city, current_time) in dp:
                return dp[(mask, current_city, current_time)]
            
            min_cost = float('inf')
            
            for next_city in range(n):
                if not (mask & (1 << next_city)):
                    travel_time = distances[current_city][next_city]
                    arrival_time = current_time + travel_time
                    earliest, latest = time_windows[next_city]
                    
                    # Check time window constraint
                    if earliest <= arrival_time <= latest:
                        new_mask = mask | (1 << next_city)
                        cost = travel_time + solve(new_mask, next_city, arrival_time)
                        min_cost = min(min_cost, cost)
            
            dp[(mask, current_city, current_time)] = min_cost
            return min_cost
        
        # Start at city 0 at time 0
        earliest_start, latest_start = time_windows[0]
        if earliest_start > 0:
            return -1
        
        result = solve(1, 0, 0)
        return result if result != float('inf') else -1
    
    def tsp_multiple_salesmen(self, distances: List[List[int]], num_salesmen: int) -> int:
        """
        Multiple Travelling Salesmen Problem
        
        Args:
            distances: Distance matrix
            num_salesmen: Number of salesmen
        
        Returns:
            Minimum total cost for all salesmen
        """
        n = len(distances)
        if num_salesmen >= n:
            return 0
        
        # dp[mask][salesmen_used] = min cost to visit cities in mask using salesmen_used salesmen
        dp = {}
        
        def solve(mask: int, salesmen_used: int) -> int:
            if mask == (1 << n) - 1:
                return 0 if salesmen_used <= num_salesmen else float('inf')
            
            if salesmen_used > num_salesmen:
                return float('inf')
            
            if (mask, salesmen_used) in dp:
                return dp[(mask, salesmen_used)]
            
            min_cost = float('inf')
            
            # Try starting a new salesman from city 0 to visit unvisited cities
            for subset_mask in range(1, (1 << n)):
                if (subset_mask & mask) == 0:  # No overlap with visited cities
                    if subset_mask & 1:  # Must include starting city 0
                        # Calculate cost for this salesman's tour
                        tour_cost = calculate_subset_tour_cost(distances, subset_mask)
                        if tour_cost != float('inf'):
                            new_mask = mask | subset_mask
                            remaining_cost = solve(new_mask, salesmen_used + 1)
                            min_cost = min(min_cost, tour_cost + remaining_cost)
            
            dp[(mask, salesmen_used)] = min_cost
            return min_cost
        
        return solve(1, 0)  # Start with city 0 visited, 0 salesmen used

def calculate_subset_tour_cost(distances: List[List[int]], subset_mask: int) -> int:
    """Helper function to calculate TSP cost for a subset of cities"""
    cities = []
    for i in range(len(distances)):
        if subset_mask & (1 << i):
            cities.append(i)
    
    if len(cities) <= 1:
        return 0
    
    n = len(cities)
    city_to_idx = {city: i for i, city in enumerate(cities)}
    
    # Create distance matrix for subset
    subset_distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            subset_distances[i][j] = distances[cities[i]][cities[j]]
    
    # Solve TSP for subset
    dp = {}
    
    def solve(mask: int, current: int) -> int:
        if mask == (1 << n) - 1:
            return subset_distances[current][0]
        
        if (mask, current) in dp:
            return dp[(mask, current)]
        
        min_cost = float('inf')
        for next_city in range(n):
            if not (mask & (1 << next_city)):
                new_mask = mask | (1 << next_city)
                cost = subset_distances[current][next_city] + solve(new_mask, next_city)
                min_cost = min(min_cost, cost)
        
        dp[(mask, current)] = min_cost
        return min_cost
    
    return solve(1, 0)

# ==================== ASSIGNMENT PROBLEM ====================

class AssignmentProblem:
    """
    Assignment Problem using Bitmasking
    
    Assign n workers to n tasks with minimum cost using bitmask DP.
    """
    
    def min_cost_assignment(self, costs: List[List[int]]) -> int:
        """
        Find minimum cost to assign all workers to tasks
        
        Time Complexity: O(n * 2^n)
        Space Complexity: O(2^n)
        
        Args:
            costs: costs[i][j] = cost of assigning worker i to task j
        
        Returns:
            Minimum total assignment cost
        """
        n = len(costs)
        if n == 0:
            return 0
        
        # dp[mask] = minimum cost to assign tasks in mask
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            # Count number of assigned tasks (worker index)
            worker = bin(mask).count('1')
            
            if worker >= n:
                continue
            
            # Try assigning current worker to each unassigned task
            for task in range(n):
                if not (mask & (1 << task)):
                    new_mask = mask | (1 << task)
                    dp[new_mask] = min(dp[new_mask], 
                                      dp[mask] + costs[worker][task])
        
        return dp[(1 << n) - 1]
    
    def assignment_with_solution(self, cost_matrix: List[List[int]]) -> Tuple[int, List[int]]:
        """
        Find minimum cost assignment and the actual assignment
        
        Args:
            cost_matrix: Cost matrix
        
        Returns:
            Tuple of (min_cost, assignment) where assignment[i] is task assigned to worker i
        """
        n = len(cost_matrix)
        if n == 0:
            return 0, []
        
        dp = [float('inf')] * (1 << n)
        parent = [-1] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            worker = bin(mask).count('1')
            if worker >= n:
                continue
            
            for task in range(n):
                if not (mask & (1 << task)):
                    new_mask = mask | (1 << task)
                    cost = dp[mask] + cost_matrix[worker][task]
                    
                    if cost < dp[new_mask]:
                        dp[new_mask] = cost
                        parent[new_mask] = task
        
        # Reconstruct assignment
        assignment = [-1] * n
        mask = (1 << n) - 1
        
        for worker in range(n - 1, -1, -1):
            task = parent[mask]
            assignment[worker] = task
            mask ^= (1 << task)
        
        return dp[(1 << n) - 1], assignment
    
    def assignment_with_constraints(self, cost_matrix: List[List[int]], 
                                  forbidden: Set[Tuple[int, int]]) -> int:
        """
        Assignment problem with forbidden worker-task pairs
        
        Args:
            cost_matrix: Cost matrix
            forbidden: Set of (worker, task) pairs that are forbidden
        
        Returns:
            Minimum cost assignment respecting constraints
        """
        n = len(cost_matrix)
        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('inf'):
                continue
            
            worker = bin(mask).count('1')
            if worker >= n:
                continue
            
            for task in range(n):
                if not (mask & (1 << task)) and (worker, task) not in forbidden:
                    new_mask = mask | (1 << task)
                    dp[new_mask] = min(dp[new_mask], 
                                     dp[mask] + cost_matrix[worker][task])
        
        return dp[(1 << n) - 1] if dp[(1 << n) - 1] != float('inf') else -1
    
    def max_assignment_profit(self, profit_matrix: List[List[int]]) -> int:
        """
        Maximum profit assignment (maximize instead of minimize)
        
        Args:
            profit_matrix: Profit matrix
        
        Returns:
            Maximum total assignment profit
        """
        n = len(profit_matrix)
        dp = [float('-inf')] * (1 << n)
        dp[0] = 0
        
        for mask in range(1 << n):
            if dp[mask] == float('-inf'):
                continue
            
            worker = bin(mask).count('1')
            if worker >= n:
                continue
            
            for task in range(n):
                if not (mask & (1 << task)):
                    new_mask = mask | (1 << task)
                    dp[new_mask] = max(dp[new_mask], 
                                     dp[mask] + profit_matrix[worker][task])
        
        return dp[(1 << n) - 1]

# ==================== GRID COVERING PROBLEMS ====================

class GridCoveringProblems:
    """
    Grid covering problems using bitmasking
    
    Count ways to cover grid with tiles or other constraints.
    """
    
    def count_ways_to_tile(self, n: int, m: int) -> int:
        """
        Count ways to tile n x m grid with 1 x 2 dominoes
        
        Time Complexity: O(n * 2^m * 2^m)
        Space Complexity: O(2^m)
        
        Args:
            n: Number of rows
            m: Number of columns
        
        Returns:
            Number of ways to tile the grid
        """
        if n == 0 or m == 0:
            return 0
        
        # dp[mask] = ways to fill current column with given mask
        dp = [0] * (1 << m)
        dp[0] = 1
        
        for col in range(n):
            new_dp = [0] * (1 << m)
            
            for mask in range(1 << m):
                if dp[mask] == 0:
                    continue
                
                self._fill_column(0, mask, 0, new_dp, dp[mask], m)
            
            dp = new_dp
        
        return dp[0]
    
    def _fill_column(self, pos: int, mask: int, next_mask: int, 
                    new_dp: List[int], ways: int, m: int):
        """
        Helper function to fill column for tiling
        """
        if pos == m:  # Reached end of column
            new_dp[next_mask] += ways
            return
        
        if mask & (1 << pos):  # Position already filled
            self._fill_column(pos + 1, mask, next_mask, new_dp, ways, m)
        else:  # Position empty, need to fill
            # Place vertical domino
            self._fill_column(pos + 1, mask | (1 << pos), 
                            next_mask | (1 << pos), new_dp, ways, m)
            
            # Place horizontal domino (if next position is also empty)
            if pos + 1 < m and not (mask & (1 << (pos + 1))):
                self._fill_column(pos + 2, mask | (1 << pos) | (1 << (pos + 1)), 
                                next_mask, new_dp, ways, m)

# ==================== SET PARTITIONING PROBLEMS ====================

class SetPartitioningProblems:
    """
    Set partitioning problems using bitmasking
    
    Various problems involving partitioning sets optimally.
    """
    
    def min_partition_difference(self, nums: List[int]) -> int:
        """
        Minimum difference between two subset sums
        
        Args:
            nums: Array of positive integers
        
        Returns:
            Minimum possible difference between subset sums
        """
        total_sum = sum(nums)
        n = len(nums)
        
        # dp[mask] = True if subset sum represented by mask is possible
        dp = [False] * (1 << n)
        dp[0] = True
        
        subset_sums = set([0])
        
        for mask in range(1 << n):
            if not dp[mask]:
                continue
            
            current_sum = sum(nums[i] for i in range(n) if mask & (1 << i))
            subset_sums.add(current_sum)
            
            for i in range(n):
                if not (mask & (1 << i)):
                    new_mask = mask | (1 << i)
                    dp[new_mask] = True
        
        min_diff = float('inf')
        for sum1 in subset_sums:
            sum2 = total_sum - sum1
            min_diff = min(min_diff, abs(sum1 - sum2))
        
        return min_diff

# ==================== PERFORMANCE ANALYSIS ====================

def performance_comparison():
    """Compare performance of different bitmask DP approaches"""
    print("=== Bitmask DP Performance Analysis ===\n")
    
    import random
    
    # Test TSP with different sizes
    print("TSP Performance:")
    for n in [4, 6, 8]:
        # Generate random distance matrix
        distances = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = random.randint(1, 100)
        
        tsp = TravellingSalesmanProblem()
        
        start_time = time.time()
        min_cost = tsp.tsp_min_cost(distances)
        time_taken = time.time() - start_time
        
        print(f"  {n} cities: cost={min_cost}, time={time_taken:.6f}s")
    
    print("\nAssignment Problem Performance:")
    assignment = AssignmentProblem()
    
    for n in [4, 6, 8]:
        cost_matrix = [[random.randint(1, 50) for _ in range(n)] for _ in range(n)]
        
        start_time = time.time()
        min_cost = assignment.min_cost_assignment(cost_matrix)
        time_taken = time.time() - start_time
        
        print(f"  {n}x{n} assignment: cost={min_cost}, time={time_taken:.6f}s")
    
    print("\nGrid Tiling Performance:")
    grid = GridCoveringProblems()
    
    test_cases = [(3, 4), (4, 4), (3, 6)]
    for n, m in test_cases:
        start_time = time.time()
        ways = grid.count_ways_to_tile(n, m)
        time_taken = time.time() - start_time
        
        print(f"  {n}x{m} tiling: ways={ways}, time={time_taken:.6f}s")

# ==================== EXAMPLE USAGE AND TESTING ====================

if __name__ == "__main__":
    print("=== Bitmasking DP Demo ===\n")
    
    # TSP Problems
    print("1. Travelling Salesman Problem:")
    tsp = TravellingSalesmanProblem()
    
    distances = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    min_cost = tsp.tsp_min_cost(distances)
    cost_with_path, path = tsp.tsp_with_path(distances)
    
    print(f"  Minimum TSP cost: {min_cost}")
    print(f"  TSP path: {path} with cost {cost_with_path}")
    print()
    
    # Assignment Problems
    print("2. Assignment Problems:")
    assignment = AssignmentProblem()
    
    costs = [
        [9, 2, 7, 8],
        [6, 4, 3, 7],
        [5, 8, 1, 8],
        [7, 6, 9, 4]
    ]
    
    min_cost = assignment.min_cost_assignment(costs)
    print(f"  Minimum assignment cost: {min_cost}")
    print()
    
    # Grid Covering Problems
    print("3. Grid Covering Problems:")
    grid = GridCoveringProblems()
    
    ways_3x4 = grid.count_ways_to_tile(3, 4)
    ways_2x3 = grid.count_ways_to_tile(2, 3)
    print(f"  Ways to tile 3x4 grid: {ways_3x4}")
    print(f"  Ways to tile 2x3 grid: {ways_2x3}")
    print()
    
    # Set Partitioning Problems
    print("4. Set Partitioning Problems:")
    partition = SetPartitioningProblems()
    
    nums = [1, 5, 11, 5]
    min_diff = partition.min_partition_difference(nums)
    
    print(f"  Array: {nums}")
    print(f"  Minimum partition difference: {min_diff}")
    print()
    
    # Performance comparison
    performance_comparison()
    
    # Pattern Recognition Guide
    print("\n=== Bitmask DP Pattern Recognition ===")
    print("When to Use Bitmask DP:")
    print("  1. Small number of elements (typically n ≤ 20)")
    print("  2. Need to track subsets or combinations")
    print("  3. State depends on which elements are included/excluded")
    print("  4. Optimization over all possible selections")
    
    print("\nCommon Bitmask DP Patterns:")
    print("  1. TSP: dp[mask][last] = min cost to visit cities in mask, ending at last")
    print("  2. Assignment: dp[mask] = min cost to assign tasks in mask")
    print("  3. Subset selection: dp[mask] = optimal value for subset mask")
    print("  4. Partitioning: dp[mask] = ways to partition subset mask")
    
    print("\nBitmask Operations:")
    print("  1. Check bit: mask & (1 << i)")
    print("  2. Set bit: mask | (1 << i)")
    print("  3. Clear bit: mask & ~(1 << i)")
    print("  4. Count bits: bin(mask).count('1')")
    print("  5. Iterate subsets: for sub in range(mask): if (sub & mask) == sub")
    
    print("\nOptimization Techniques:")
    print("  1. Precompute subset properties (sum, value, etc.)")
    print("  2. Use subset enumeration for faster transitions")
    print("  3. Profile iterating vs. direct computation")
    print("  4. Consider state compression for larger problems")
    
    print("\nReal-world Applications:")
    print("  1. Resource allocation with constraints")
    print("  2. Scheduling with dependencies")
    print("  3. Network design and optimization")
    print("  4. Combinatorial auctions")
    print("  5. Feature selection in machine learning")
    print("  6. Circuit design and optimization")
    
    print("\nCommon Pitfalls:")
    print("  1. Exponential space complexity (2^n)")
    print("  2. Integer overflow with large n")
    print("  3. Incorrect bit manipulation")
    print("  4. Not considering all possible transitions")
    print("  5. Inefficient subset enumeration")
    
    print("\n=== Demo Complete ===") 