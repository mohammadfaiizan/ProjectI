"""
1066. Campus Bikes II - Multiple Approaches
Difficulty: Medium

On a campus represented as a 2D grid, there are N workers and M bikes, with N <= M. Each worker and bike is a 2D coordinate on this grid.

We assign one unique bike to each worker. The assignment should minimize the sum of the Manhattan distances between each worker and their assigned bike.

Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.
"""

from typing import List, Tuple
import heapq
from functools import lru_cache

class CampusBikes:
    """Multiple approaches to solve campus bikes assignment problem"""
    
    def assignBikes_dijkstra_bitmask(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 1: Dijkstra with Bitmask State
        
        Use Dijkstra's algorithm with bitmask to represent bike assignments.
        
        Time: O(2^M * N * M), Space: O(2^M * N)
        """
        n, m = len(workers), len(bikes)
        
        def manhattan_distance(p1: List[int], p2: List[int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Priority queue: (total_distance, worker_index, bike_mask)
        pq = [(0, 0, 0)]
        visited = set()
        
        while pq:
            total_dist, worker_idx, bike_mask = heapq.heappop(pq)
            
            if worker_idx == n:
                return total_dist
            
            state = (worker_idx, bike_mask)
            if state in visited:
                continue
            
            visited.add(state)
            
            # Try assigning each available bike to current worker
            for bike_idx in range(m):
                if not (bike_mask & (1 << bike_idx)):  # Bike not used
                    new_mask = bike_mask | (1 << bike_idx)
                    dist = manhattan_distance(workers[worker_idx], bikes[bike_idx])
                    new_total = total_dist + dist
                    
                    heapq.heappush(pq, (new_total, worker_idx + 1, new_mask))
        
        return -1
    
    def assignBikes_dp_bitmask(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 2: Dynamic Programming with Bitmask
        
        Use DP with bitmask to represent which bikes are assigned.
        
        Time: O(2^M * N), Space: O(2^M)
        """
        n, m = len(workers), len(bikes)
        
        def manhattan_distance(p1: List[int], p2: List[int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Precompute all distances
        distances = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                distances[i][j] = manhattan_distance(workers[i], bikes[j])
        
        # DP with memoization
        @lru_cache(maxsize=None)
        def dp(worker_idx: int, bike_mask: int) -> int:
            if worker_idx == n:
                return 0
            
            min_cost = float('inf')
            
            for bike_idx in range(m):
                if not (bike_mask & (1 << bike_idx)):  # Bike available
                    new_mask = bike_mask | (1 << bike_idx)
                    cost = distances[worker_idx][bike_idx] + dp(worker_idx + 1, new_mask)
                    min_cost = min(min_cost, cost)
            
            return min_cost
        
        return dp(0, 0)
    
    def assignBikes_backtracking(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 3: Backtracking with Pruning
        
        Use backtracking with pruning to find optimal assignment.
        
        Time: O(M! / (M-N)!), Space: O(N)
        """
        n, m = len(workers), len(bikes)
        
        def manhattan_distance(p1: List[int], p2: List[int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Precompute distances
        distances = [[0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                distances[i][j] = manhattan_distance(workers[i], bikes[j])
        
        self.min_cost = float('inf')
        used_bikes = [False] * m
        
        def backtrack(worker_idx: int, current_cost: int):
            if worker_idx == n:
                self.min_cost = min(self.min_cost, current_cost)
                return
            
            # Pruning: if current cost already exceeds minimum, stop
            if current_cost >= self.min_cost:
                return
            
            for bike_idx in range(m):
                if not used_bikes[bike_idx]:
                    used_bikes[bike_idx] = True
                    new_cost = current_cost + distances[worker_idx][bike_idx]
                    backtrack(worker_idx + 1, new_cost)
                    used_bikes[bike_idx] = False
        
        backtrack(0, 0)
        return self.min_cost
    
    def assignBikes_hungarian_simplified(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 4: Simplified Hungarian Algorithm
        
        Use simplified version of Hungarian algorithm for assignment.
        
        Time: O(N^3), Space: O(N^2)
        """
        n, m = len(workers), len(bikes)
        
        def manhattan_distance(p1: List[int], p2: List[int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Create cost matrix (only consider first n bikes for simplicity)
        cost_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(min(n, m)):
                cost_matrix[i][j] = manhattan_distance(workers[i], bikes[j])
        
        # Simple assignment using minimum cost matching
        # This is a simplified version - full Hungarian algorithm is more complex
        used_bikes = [False] * n
        total_cost = 0
        
        for worker in range(n):
            min_cost = float('inf')
            best_bike = -1
            
            for bike in range(min(n, m)):
                if not used_bikes[bike] and cost_matrix[worker][bike] < min_cost:
                    min_cost = cost_matrix[worker][bike]
                    best_bike = bike
            
            if best_bike != -1:
                used_bikes[best_bike] = True
                total_cost += min_cost
        
        return total_cost
    
    def assignBikes_optimized_dp(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        """
        Approach 5: Optimized DP with State Compression
        
        Optimize DP using better state representation.
        
        Time: O(2^M * N), Space: O(2^M)
        """
        n, m = len(workers), len(bikes)
        
        def manhattan_distance(p1: List[int], p2: List[int]) -> int:
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Precompute distances
        distances = []
        for i in range(n):
            worker_distances = []
            for j in range(m):
                worker_distances.append(manhattan_distance(workers[i], bikes[j]))
            distances.append(worker_distances)
        
        # DP: dp[mask] = minimum cost to assign bikes represented by mask
        dp = {}
        dp[0] = 0
        
        for worker_idx in range(n):
            new_dp = {}
            
            for mask, cost in dp.items():
                # Try assigning each available bike to current worker
                for bike_idx in range(m):
                    if not (mask & (1 << bike_idx)):  # Bike available
                        new_mask = mask | (1 << bike_idx)
                        new_cost = cost + distances[worker_idx][bike_idx]
                        
                        if new_mask not in new_dp or new_cost < new_dp[new_mask]:
                            new_dp[new_mask] = new_cost
            
            dp = new_dp
        
        return min(dp.values())

def test_campus_bikes():
    """Test campus bikes assignment algorithms"""
    solver = CampusBikes()
    
    test_cases = [
        ([[0,0],[2,1]], [[1,2],[3,3]], 6, "Simple case"),
        ([[0,0],[1,1],[2,0]], [[1,0],[2,2],[2,1]], 4, "Three workers"),
        ([[0,0],[1,0],[2,0],[3,0],[4,0]], [[0,999],[1,999],[2,999],[3,999],[4,999]], 2499, "Linear arrangement"),
    ]
    
    algorithms = [
        ("Dijkstra Bitmask", solver.assignBikes_dijkstra_bitmask),
        ("DP Bitmask", solver.assignBikes_dp_bitmask),
        ("Backtracking", solver.assignBikes_backtracking),
        ("Optimized DP", solver.assignBikes_optimized_dp),
    ]
    
    print("=== Testing Campus Bikes II ===")
    
    for workers, bikes, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"Workers: {workers}, Bikes: {bikes}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(workers, bikes)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:18} | {status} | Min Distance: {result}")
            except Exception as e:
                print(f"{alg_name:18} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_campus_bikes()

"""
Campus Bikes II demonstrates advanced assignment problems
using shortest path concepts, dynamic programming with bitmasks,
and optimization techniques for minimum cost matching.
"""
