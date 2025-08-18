"""
1615. Maximal Network Rank
Difficulty: Medium

Problem:
There is an infrastructure of n cities with some number of roads connecting these cities. 
Each roads[i] = [ai, bi] indicates that there is a bidirectional road between cities ai and bi.

The network rank of two different cities is defined as the total number of directly 
connected roads to either city. If a road is directly connected to both cities, 
it is only counted once.

Return the maximal network rank of the entire infrastructure.

Examples:
Input: n = 4, roads = [[0,1],[0,3],[1,2],[1,3]]
Output: 4
Explanation: The network rank of cities 0 and 1 is 4 as there are 4 roads that 
are connected to either 0 or 1. The road between 0 and 1 is only counted once.

Input: n = 5, roads = [[0,1],[0,3],[1,2],[1,3],[2,3],[2,4]]
Output: 5

Input: n = 8, roads = [[0,1],[1,2],[2,3],[2,4],[5,6],[5,7]]
Output: 5

Constraints:
- 2 <= n <= 100
- 0 <= roads.length <= n * (n - 1) / 2
- roads[i].length == 2
- 0 <= ai, bi <= n-1
- ai != bi
- Each pair of cities has at most one road connecting them
"""

from typing import List, Set
from collections import defaultdict

class Solution:
    def maximalNetworkRank_approach1_brute_force(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 1: Brute force - Check all pairs
        
        For each pair of cities, calculate their network rank:
        rank(i, j) = degree(i) + degree(j) - (1 if connected else 0)
        
        Time: O(N² + E) where E = number of roads
        Space: O(N + E)
        """
        # Build adjacency set for O(1) edge checking
        adj = defaultdict(set)
        degree = [0] * n
        
        for a, b in roads:
            adj[a].add(b)
            adj[b].add(a)
            degree[a] += 1
            degree[b] += 1
        
        max_rank = 0
        
        # Check all pairs of cities
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate network rank for cities i and j
                rank = degree[i] + degree[j]
                
                # Subtract 1 if there's a direct connection
                if j in adj[i]:
                    rank -= 1
                
                max_rank = max(max_rank, rank)
        
        return max_rank
    
    def maximalNetworkRank_approach2_optimized_degree_sorting(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 2: Optimization using degree sorting
        
        Key insight: The maximum rank is likely to involve cities with high degrees.
        Sort cities by degree and check high-degree pairs first.
        
        Time: O(N log N + E)
        Space: O(N + E)
        """
        # Build graph representation
        adj = defaultdict(set)
        degree = [0] * n
        
        for a, b in roads:
            adj[a].add(b)
            adj[b].add(a)
            degree[a] += 1
            degree[b] += 1
        
        # Sort cities by degree (descending)
        cities_by_degree = sorted(range(n), key=lambda x: degree[x], reverse=True)
        
        max_rank = 0
        
        # Check pairs starting with highest degree cities
        for i in range(n):
            for j in range(i + 1, n):
                city1 = cities_by_degree[i]
                city2 = cities_by_degree[j]
                
                rank = degree[city1] + degree[city2]
                
                # Subtract 1 if directly connected
                if city2 in adj[city1]:
                    rank -= 1
                
                max_rank = max(max_rank, rank)
                
                # Early termination optimization
                # If remaining cities can't beat current max, stop
                if i > 0 and degree[cities_by_degree[i]] + degree[cities_by_degree[0]] <= max_rank:
                    break
        
        return max_rank
    
    def maximalNetworkRank_approach3_smart_pruning(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 3: Smart pruning with mathematical bounds
        
        Use upper bounds to prune unnecessary computations.
        
        Time: O(N² + E) with better constants
        Space: O(N + E)
        """
        # Build graph
        adj = [set() for _ in range(n)]
        degree = [0] * n
        
        for a, b in roads:
            adj[a].add(b)
            adj[b].add(a)
            degree[a] += 1
            degree[b] += 1
        
        # Find max and second max degrees for upper bound estimation
        max_degree = max(degree) if degree else 0
        second_max = 0
        for d in degree:
            if d < max_degree:
                second_max = max(second_max, d)
        
        max_rank = 0
        
        for i in range(n):
            # Pruning: if degree[i] + max_degree <= max_rank, skip
            if degree[i] + max_degree <= max_rank:
                continue
                
            for j in range(i + 1, n):
                # Pruning: if current pair can't beat max_rank, skip
                if degree[i] + degree[j] <= max_rank:
                    continue
                
                rank = degree[i] + degree[j]
                
                if j in adj[i]:
                    rank -= 1
                
                max_rank = max(max_rank, rank)
        
        return max_rank
    
    def maximalNetworkRank_approach4_edge_contribution(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 4: Edge contribution analysis
        
        Think of each edge contributing to multiple city pairs.
        Analyze which pairs benefit most from edge contributions.
        
        Time: O(N² + E)
        Space: O(N + E)
        """
        if not roads:
            return 0
        
        # Build adjacency matrix for O(1) lookup
        connected = [[False] * n for _ in range(n)]
        degree = [0] * n
        
        for a, b in roads:
            connected[a][b] = connected[b][a] = True
            degree[a] += 1
            degree[b] += 1
        
        max_rank = 0
        
        # Calculate rank for each pair
        for i in range(n):
            for j in range(i + 1, n):
                rank = degree[i] + degree[j]
                
                # Subtract overlap if directly connected
                if connected[i][j]:
                    rank -= 1
                
                max_rank = max(max_rank, rank)
        
        return max_rank
    
    def maximalNetworkRank_approach5_mathematical_analysis(self, n: int, roads: List[List[int]]) -> int:
        """
        Approach 5: Mathematical analysis approach
        
        Analyze the problem from a mathematical perspective.
        The maximum rank involves finding the pair with maximum degree sum,
        considering potential overlap.
        
        Time: O(N² + E)
        Space: O(N + E)
        """
        degree = [0] * n
        edge_set = set()
        
        for a, b in roads:
            degree[a] += 1
            degree[b] += 1
            edge_set.add((min(a, b), max(a, b)))
        
        max_rank = 0
        
        # Mathematical optimization: check pairs in order of potential rank
        degree_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                potential_rank = degree[i] + degree[j]
                degree_pairs.append((potential_rank, i, j))
        
        # Sort by potential rank (descending)
        degree_pairs.sort(reverse=True)
        
        for potential_rank, i, j in degree_pairs:
            # Early termination: if potential rank can't beat current max, stop
            if potential_rank <= max_rank:
                break
            
            actual_rank = potential_rank
            if (min(i, j), max(i, j)) in edge_set:
                actual_rank -= 1
            
            max_rank = max(max_rank, actual_rank)
        
        return max_rank

def test_maximal_network_rank():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, roads, expected)
        (4, [[0,1],[0,3],[1,2],[1,3]], 4),
        (5, [[0,1],[0,3],[1,2],[1,3],[2,3],[2,4]], 5),
        (8, [[0,1],[1,2],[2,3],[2,4],[5,6],[5,7]], 5),
        (2, [[0,1]], 2),  # Minimal case
        (3, [], 0),       # No roads
        (4, [[0,1],[2,3]], 2),  # Disconnected pairs
        (3, [[0,1],[0,2],[1,2]], 4),  # Complete triangle
    ]
    
    approaches = [
        ("Brute Force", solution.maximalNetworkRank_approach1_brute_force),
        ("Degree Sorting", solution.maximalNetworkRank_approach2_optimized_degree_sorting),
        ("Smart Pruning", solution.maximalNetworkRank_approach3_smart_pruning),
        ("Edge Contribution", solution.maximalNetworkRank_approach4_edge_contribution),
        ("Mathematical Analysis", solution.maximalNetworkRank_approach5_mathematical_analysis),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, roads, expected) in enumerate(test_cases):
            result = func(n, roads)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, roads={roads}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_network_rank_calculation():
    """Demonstrate how network rank is calculated"""
    print("\n=== Network Rank Calculation Demo ===")
    
    n = 4
    roads = [[0,1],[0,3],[1,2],[1,3]]
    
    print(f"Cities: {list(range(n))}")
    print(f"Roads: {roads}")
    
    # Build adjacency and calculate degrees
    adj = defaultdict(set)
    degree = [0] * n
    
    for a, b in roads:
        adj[a].add(b)
        adj[b].add(a)
        degree[a] += 1
        degree[b] += 1
    
    print(f"\nDegrees: {degree}")
    print(f"Adjacency:")
    for i in range(n):
        print(f"  City {i}: connected to {sorted(adj[i])}")
    
    print(f"\nNetwork rank calculation for all pairs:")
    max_rank = 0
    best_pair = None
    
    for i in range(n):
        for j in range(i + 1, n):
            rank = degree[i] + degree[j]
            connected = j in adj[i]
            if connected:
                rank -= 1
            
            print(f"  Cities ({i}, {j}): degree_sum={degree[i] + degree[j]}, " +
                  f"connected={connected}, rank={rank}")
            
            if rank > max_rank:
                max_rank = rank
                best_pair = (i, j)
    
    print(f"\nMaximal network rank: {max_rank}")
    print(f"Best pair: {best_pair}")

def analyze_graph_properties():
    """Analyze properties that affect network rank"""
    print("\n=== Graph Properties Analysis ===")
    
    examples = [
        ("Complete graph K4", 4, [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]),
        ("Star graph", 4, [[0,1],[0,2],[0,3]]),
        ("Path graph", 4, [[0,1],[1,2],[2,3]]),
        ("Disconnected", 4, [[0,1],[2,3]]),
    ]
    
    solution = Solution()
    
    for name, n, roads in examples:
        rank = solution.maximalNetworkRank_approach1_brute_force(n, roads)
        
        # Calculate average degree
        degree_sum = sum(len(adj) for road in roads for adj in [road])
        avg_degree = degree_sum / n if n > 0 else 0
        
        print(f"{name}:")
        print(f"  Roads: {roads}")
        print(f"  Max network rank: {rank}")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Density: {len(roads) / (n * (n-1) / 2):.2f}")

if __name__ == "__main__":
    test_maximal_network_rank()
    demonstrate_network_rank_calculation()
    analyze_graph_properties()

"""
Graph Theory Concepts:
1. Network Centrality and Rank
2. Degree Distribution Analysis  
3. Graph Density and Connectivity
4. Optimization through Mathematical Bounds

Key Insights:
- Network rank measures combined connectivity of city pairs
- High-degree cities are likely to form high-rank pairs
- Direct connections between high-degree cities reduce their rank
- Problem has O(N²) pairs but can be optimized with smart pruning

Mathematical Analysis:
- Maximum possible rank = 2 * (total_edges) when cities share no edges
- Minimum rank for connected pair = degree_sum - 1
- Optimal strategy: balance high individual degrees with low overlap

Real-world Applications:
- Transportation network analysis
- Communication network design
- Social network influence measurement
- Infrastructure resilience assessment
- Network capacity planning
"""
