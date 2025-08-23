"""
TSP Advanced Algorithms - Comprehensive Implementation
Difficulty: Hard

The Traveling Salesman Problem (TSP) is one of the most famous NP-hard problems.
This file implements various advanced algorithms for solving TSP, from exact
algorithms to sophisticated approximations and heuristics.

Key Concepts:
1. Held-Karp Dynamic Programming
2. Branch and Bound with MST Lower Bounds
3. Christofides Algorithm (2-approximation)
4. Lin-Kernighan Heuristic
5. Genetic Algorithm Approach
6. Simulated Annealing
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import heapq
import random
import math
import itertools

class TSPAdvancedAlgorithms:
    """Advanced algorithms for Traveling Salesman Problem"""
    
    def __init__(self):
        self.n = 0
        self.dist = []
        self.cities = []
    
    def set_distance_matrix(self, distance_matrix: List[List[float]]):
        """Set distance matrix for TSP instance"""
        self.dist = distance_matrix
        self.n = len(distance_matrix)
        self.cities = list(range(self.n))
    
    def tsp_held_karp_dp(self) -> Tuple[float, List[int]]:
        """
        Approach 1: Held-Karp Dynamic Programming Algorithm
        
        Exact algorithm using dynamic programming with bitmasks.
        
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
        
        # dp[mask][i] = minimum cost to visit cities in mask, ending at city i
        dp = {}
        parent = {}
        
        # Initialize: starting from city 0
        for i in range(1, self.n):
            dp[(1 << i) | 1, i] = self.dist[0][i]
            parent[(1 << i) | 1, i] = 0
        
        # Fill DP table
        for mask_size in range(2, self.n):
            for mask in range(1, 1 << self.n):
                if bin(mask).count('1') != mask_size + 1:
                    continue
                if not (mask & 1):  # Must include city 0
                    continue
                
                for i in range(1, self.n):
                    if not (mask & (1 << i)):
                        continue
                    
                    prev_mask = mask ^ (1 << i)
                    min_cost = float('inf')
                    best_prev = -1
                    
                    for j in range(1, self.n):
                        if i == j or not (prev_mask & (1 << j)):
                            continue
                        
                        if (prev_mask, j) in dp:
                            cost = dp[(prev_mask, j)] + self.dist[j][i]
                            if cost < min_cost:
                                min_cost = cost
                                best_prev = j
                    
                    if best_prev != -1:
                        dp[(mask, i)] = min_cost
                        parent[(mask, i)] = best_prev
        
        # Find minimum cost to return to start
        full_mask = (1 << self.n) - 1
        min_cost = float('inf')
        last_city = -1
        
        for i in range(1, self.n):
            if (full_mask, i) in dp:
                cost = dp[(full_mask, i)] + self.dist[i][0]
                if cost < min_cost:
                    min_cost = cost
                    last_city = i
        
        # Reconstruct path
        if last_city == -1:
            return float('inf'), []
        
        path = self._reconstruct_path_dp(parent, full_mask, last_city)
        return min_cost, path
    
    def tsp_branch_and_bound_mst(self) -> Tuple[float, List[int]]:
        """
        Approach 2: Branch and Bound with MST Lower Bound
        
        Use MST-based lower bounds for pruning in branch and bound.
        
        Time: Exponential with pruning, Space: O(n)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
        
        self.best_cost = float('inf')
        self.best_path = []
        
        def mst_lower_bound(remaining: Set[int], current_cost: float, last_city: int) -> float:
            """Calculate MST-based lower bound"""
            if len(remaining) <= 1:
                return current_cost
            
            # MST of remaining cities
            mst_cost = self._minimum_spanning_tree_cost(remaining)
            
            # Add minimum edge from last city to remaining
            min_to_remaining = min(self.dist[last_city][city] for city in remaining)
            
            # Add minimum edge from remaining back to start
            min_to_start = min(self.dist[city][0] for city in remaining)
            
            return current_cost + min_to_remaining + mst_cost + min_to_start
        
        def branch_and_bound(path: List[int], remaining: Set[int], current_cost: float):
            if not remaining:
                # Complete tour
                total_cost = current_cost + self.dist[path[-1]][path[0]]
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_path = path[:]
                return
            
            # Pruning
            lower_bound = mst_lower_bound(remaining, current_cost, path[-1])
            if lower_bound >= self.best_cost:
                return
            
            # Branch to remaining cities
            for next_city in remaining:
                new_cost = current_cost + self.dist[path[-1]][next_city]
                if new_cost < self.best_cost:
                    path.append(next_city)
                    remaining.remove(next_city)
                    
                    branch_and_bound(path, remaining, new_cost)
                    
                    path.pop()
                    remaining.add(next_city)
        
        # Start branch and bound
        remaining = set(range(1, self.n))
        branch_and_bound([0], remaining, 0.0)
        
        return self.best_cost, self.best_path
    
    def tsp_christofides_approximation(self) -> Tuple[float, List[int]]:
        """
        Approach 3: Christofides Algorithm (2-approximation)
        
        Guaranteed 2-approximation for metric TSP.
        
        Time: O(n^3), Space: O(n^2)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
        
        # Step 1: Find MST
        mst_edges = self._minimum_spanning_tree()
        
        # Step 2: Find vertices with odd degree in MST
        degree = [0] * self.n
        for u, v, _ in mst_edges:
            degree[u] += 1
            degree[v] += 1
        
        odd_vertices = [i for i in range(self.n) if degree[i] % 2 == 1]
        
        # Step 3: Find minimum weight perfect matching on odd vertices
        matching_edges = self._minimum_weight_perfect_matching(odd_vertices)
        
        # Step 4: Combine MST and matching to form Eulerian multigraph
        multigraph = defaultdict(list)
        for u, v, _ in mst_edges:
            multigraph[u].append(v)
            multigraph[v].append(u)
        
        for u, v in matching_edges:
            multigraph[u].append(v)
            multigraph[v].append(u)
        
        # Step 5: Find Eulerian tour
        eulerian_tour = self._find_eulerian_tour(multigraph, 0)
        
        # Step 6: Convert to Hamiltonian by skipping repeated vertices
        visited = set()
        hamiltonian_path = []
        
        for city in eulerian_tour:
            if city not in visited:
                hamiltonian_path.append(city)
                visited.add(city)
        
        # Calculate tour cost
        tour_cost = sum(self.dist[hamiltonian_path[i]][hamiltonian_path[(i + 1) % len(hamiltonian_path)]]
                       for i in range(len(hamiltonian_path)))
        
        return tour_cost, hamiltonian_path
    
    def tsp_lin_kernighan_heuristic(self, max_iterations: int = 1000) -> Tuple[float, List[int]]:
        """
        Approach 4: Lin-Kernighan Heuristic
        
        Advanced local search heuristic with variable neighborhood.
        
        Time: O(n^2.2) per iteration, Space: O(n)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
        
        # Start with nearest neighbor tour
        current_tour = self._nearest_neighbor_tour()
        current_cost = self._calculate_tour_cost(current_tour)
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try 2-opt improvements
            for i in range(self.n):
                for j in range(i + 2, self.n):
                    if j == self.n - 1 and i == 0:
                        continue
                    
                    new_tour = current_tour[:]
                    new_tour[i + 1:j + 1] = reversed(new_tour[i + 1:j + 1])
                    
                    new_cost = self._calculate_tour_cost(new_tour)
                    if new_cost < current_cost:
                        current_tour = new_tour
                        current_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
            
            # Try 3-opt improvements (simplified)
            if not improved:
                for i in range(self.n):
                    for j in range(i + 2, self.n):
                        for k in range(j + 2, self.n):
                            # Try one of the 3-opt moves
                            new_tour = self._three_opt_move(current_tour, i, j, k)
                            new_cost = self._calculate_tour_cost(new_tour)
                            
                            if new_cost < current_cost:
                                current_tour = new_tour
                                current_cost = new_cost
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        return current_cost, current_tour
    
    def tsp_genetic_algorithm(self, population_size: int = 100, generations: int = 500) -> Tuple[float, List[int]]:
        """
        Approach 5: Genetic Algorithm
        
        Evolutionary approach with crossover and mutation.
        
        Time: O(generations * population_size * n), Space: O(population_size * n)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
        
        # Initialize population
        population = []
        for _ in range(population_size):
            tour = list(range(self.n))
            random.shuffle(tour[1:])  # Keep 0 as start
            population.append(tour)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [(self._calculate_tour_cost(tour), tour) for tour in population]
            fitness_scores.sort()
            
            # Select top 50% for breeding
            elite_size = population_size // 2
            elite = [tour for _, tour in fitness_scores[:elite_size]]
            
            # Create new population
            new_population = elite[:]
            
            while len(new_population) < population_size:
                # Selection
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover (Order Crossover - OX)
                child = self._order_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < 0.1:  # 10% mutation rate
                    child = self._mutate_tour(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best solution
        best_cost = float('inf')
        best_tour = None
        
        for tour in population:
            cost = self._calculate_tour_cost(tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
        
        return best_cost, best_tour
    
    def tsp_simulated_annealing(self, initial_temp: float = 1000.0, 
                               cooling_rate: float = 0.995, min_temp: float = 1.0) -> Tuple[float, List[int]]:
        """
        Approach 6: Simulated Annealing
        
        Probabilistic optimization technique inspired by metallurgy.
        
        Time: O(iterations * n), Space: O(n)
        """
        if self.n <= 1:
            return 0.0, list(range(self.n))
        
        # Start with random tour
        current_tour = list(range(self.n))
        random.shuffle(current_tour[1:])
        current_cost = self._calculate_tour_cost(current_tour)
        
        best_tour = current_tour[:]
        best_cost = current_cost
        
        temperature = initial_temp
        
        while temperature > min_temp:
            # Generate neighbor by 2-opt swap
            i, j = random.sample(range(1, self.n), 2)
            if i > j:
                i, j = j, i
            
            new_tour = current_tour[:]
            new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
            new_cost = self._calculate_tour_cost(new_tour)
            
            # Accept or reject
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_tour = new_tour
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_tour = current_tour[:]
                    best_cost = current_cost
            
            temperature *= cooling_rate
        
        return best_cost, best_tour
    
    def _reconstruct_path_dp(self, parent: Dict, mask: int, last_city: int) -> List[int]:
        """Reconstruct path from DP parent pointers"""
        path = []
        current_mask = mask
        current_city = last_city
        
        while current_city != -1:
            path.append(current_city)
            if (current_mask, current_city) not in parent:
                break
            
            next_city = parent[(current_mask, current_city)]
            current_mask ^= (1 << current_city)
            current_city = next_city
        
        path.reverse()
        return path
    
    def _minimum_spanning_tree_cost(self, vertices: Set[int]) -> float:
        """Calculate MST cost for given vertices"""
        if len(vertices) <= 1:
            return 0.0
        
        vertices_list = list(vertices)
        n = len(vertices_list)
        
        # Prim's algorithm
        in_mst = [False] * n
        key = [float('inf')] * n
        key[0] = 0.0
        
        mst_cost = 0.0
        
        for _ in range(n):
            # Find minimum key vertex not in MST
            min_key = float('inf')
            u = -1
            
            for v in range(n):
                if not in_mst[v] and key[v] < min_key:
                    min_key = key[v]
                    u = v
            
            if u == -1:
                break
            
            in_mst[u] = True
            mst_cost += key[u]
            
            # Update keys
            for v in range(n):
                if not in_mst[v]:
                    weight = self.dist[vertices_list[u]][vertices_list[v]]
                    if weight < key[v]:
                        key[v] = weight
        
        return mst_cost
    
    def _minimum_spanning_tree(self) -> List[Tuple[int, int, float]]:
        """Find MST using Prim's algorithm"""
        if self.n <= 1:
            return []
        
        in_mst = [False] * self.n
        key = [float('inf')] * self.n
        parent = [-1] * self.n
        key[0] = 0.0
        
        edges = []
        
        for _ in range(self.n):
            # Find minimum key vertex
            min_key = float('inf')
            u = -1
            
            for v in range(self.n):
                if not in_mst[v] and key[v] < min_key:
                    min_key = key[v]
                    u = v
            
            if u == -1:
                break
            
            in_mst[u] = True
            
            if parent[u] != -1:
                edges.append((parent[u], u, self.dist[parent[u]][u]))
            
            # Update keys
            for v in range(self.n):
                if not in_mst[v] and self.dist[u][v] < key[v]:
                    key[v] = self.dist[u][v]
                    parent[v] = u
        
        return edges
    
    def _minimum_weight_perfect_matching(self, vertices: List[int]) -> List[Tuple[int, int]]:
        """Find minimum weight perfect matching (simplified greedy approach)"""
        if len(vertices) % 2 != 0:
            return []
        
        vertices = vertices[:]
        matching = []
        
        while len(vertices) >= 2:
            min_weight = float('inf')
            best_pair = None
            
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    weight = self.dist[vertices[i]][vertices[j]]
                    if weight < min_weight:
                        min_weight = weight
                        best_pair = (i, j)
            
            if best_pair:
                i, j = best_pair
                matching.append((vertices[i], vertices[j]))
                # Remove in reverse order to maintain indices
                vertices.pop(max(i, j))
                vertices.pop(min(i, j))
        
        return matching
    
    def _find_eulerian_tour(self, graph: Dict, start: int) -> List[int]:
        """Find Eulerian tour using Hierholzer's algorithm"""
        # Make a copy of the graph
        g = defaultdict(list)
        for u in graph:
            g[u] = graph[u][:]
        
        tour = []
        stack = [start]
        
        while stack:
            v = stack[-1]
            if g[v]:
                u = g[v].pop()
                g[u].remove(v)
                stack.append(u)
            else:
                tour.append(stack.pop())
        
        return tour[::-1]
    
    def _nearest_neighbor_tour(self) -> List[int]:
        """Generate initial tour using nearest neighbor heuristic"""
        tour = [0]
        unvisited = set(range(1, self.n))
        
        current = 0
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.dist[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour
    
    def _calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate total cost of tour"""
        if len(tour) <= 1:
            return 0.0
        
        cost = 0.0
        for i in range(len(tour)):
            cost += self.dist[tour[i]][tour[(i + 1) % len(tour)]]
        
        return cost
    
    def _three_opt_move(self, tour: List[int], i: int, j: int, k: int) -> List[int]:
        """Perform 3-opt move (simplified version)"""
        new_tour = tour[:]
        # Simple 3-opt: reverse middle segment
        new_tour[i + 1:j + 1] = reversed(new_tour[i + 1:j + 1])
        return new_tour
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover for genetic algorithm"""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        child = [-1] * n
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer % n] = city
                pointer += 1
        
        return child
    
    def _mutate_tour(self, tour: List[int]) -> List[int]:
        """Mutate tour by swapping two random cities"""
        mutated = tour[:]
        i, j = random.sample(range(1, len(tour)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

def test_tsp_algorithms():
    """Test TSP algorithms on small instances"""
    print("=== Testing TSP Advanced Algorithms ===")
    
    # Small test instance
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    tsp = TSPAdvancedAlgorithms()
    tsp.set_distance_matrix(distance_matrix)
    
    algorithms = [
        ("Held-Karp DP", tsp.tsp_held_karp_dp),
        ("Branch & Bound", tsp.tsp_branch_and_bound_mst),
        ("Christofides", tsp.tsp_christofides_approximation),
        ("Lin-Kernighan", lambda: tsp.tsp_lin_kernighan_heuristic(100)),
        ("Genetic Algorithm", lambda: tsp.tsp_genetic_algorithm(50, 100)),
        ("Simulated Annealing", tsp.tsp_simulated_annealing),
    ]
    
    print(f"Distance Matrix:")
    for row in distance_matrix:
        print(f"  {row}")
    
    print(f"\nAlgorithm Results:")
    
    for alg_name, alg_func in algorithms:
        try:
            cost, tour = alg_func()
            print(f"{alg_name:18} | Cost: {cost:6.1f} | Tour: {tour}")
        except Exception as e:
            print(f"{alg_name:18} | ERROR: {str(e)[:40]}")

if __name__ == "__main__":
    test_tsp_algorithms()

"""
TSP Advanced Algorithms demonstrates the evolution of approaches
to one of the most famous NP-hard problems, from exact exponential
algorithms to sophisticated approximations and metaheuristics.
"""
