"""
1820. Maximum Number of Accepted Invitations
Difficulty: Medium

Problem:
There are m boys and n girls in a class attending an upcoming party.

You are given an m x n integer matrix grid, where grid[i][j] equals 1 if the ith boy can invite the jth girl to the party, and equals 0 otherwise.

A boy can invite at most one girl, and a girl can accept at most one invitation. Return the maximum number of accepted invitations.

Examples:
Input: grid = [[1,1,1],
               [1,0,1],
               [0,0,1]]
Output: 3
Explanation: The invitations are sent as follows:
- Boy 0 invites girl 0.
- Boy 1 invites girl 2.
- Boy 2 invites girl 2.
Girl 2 can only accept one invitation so boy 1 and boy 2 cannot both invite her.
We can achieve the maximum number of accepted invitations, 3, in the following way:
- Boy 0 invites girl 1.
- Boy 1 invites girl 0.
- Boy 2 invites girl 2.

Input: grid = [[1,0,1,0],
               [1,0,0,0],
               [0,0,1,0],
               [0,0,1,1]]
Output: 3

Input: grid = [[1,0,0,0],
               [0,0,0,0],
               [0,0,0,0]]
Output: 1

Constraints:
- grid.length == m
- grid[i].length == n
- 1 <= m, n <= 200
- grid[i][j] is either 0 or 1.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class Solution:
    def maximumInvitations_approach1_maximum_bipartite_matching_dfs(self, grid: List[List[int]]) -> int:
        """
        Approach 1: Maximum Bipartite Matching using DFS
        
        Use DFS-based augmenting path algorithm for maximum matching.
        
        Time: O(V * E) where V = boys+girls, E = total edges
        Space: O(V + E)
        """
        m, n = len(grid), len(grid[0])
        
        # girl_match[j] = boy matched to girl j, -1 if unmatched
        girl_match = [-1] * n
        
        def dfs_find_augmenting_path(boy, visited_girls):
            """Find augmenting path starting from unmatched boy"""
            for girl in range(n):
                if grid[boy][girl] == 1 and girl not in visited_girls:
                    visited_girls.add(girl)
                    
                    # If girl is unmatched or we can find augmenting path from her current match
                    if girl_match[girl] == -1 or dfs_find_augmenting_path(girl_match[girl], visited_girls):
                        girl_match[girl] = boy
                        return True
            
            return False
        
        matching_count = 0
        
        # Try to match each boy
        for boy in range(m):
            visited_girls = set()
            if dfs_find_augmenting_path(boy, visited_girls):
                matching_count += 1
        
        return matching_count
    
    def maximumInvitations_approach2_maximum_bipartite_matching_bfs(self, grid: List[List[int]]) -> int:
        """
        Approach 2: Maximum Bipartite Matching using BFS (Ford-Fulkerson style)
        
        Use BFS to find augmenting paths for maximum matching.
        
        Time: O(V * E)
        Space: O(V + E)
        """
        m, n = len(grid), len(grid[0])
        
        # Build bipartite graph
        boy_neighbors = [[] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    boy_neighbors[i].append(j)
        
        # Matching arrays
        boy_match = [-1] * m  # boy_match[i] = girl matched to boy i
        girl_match = [-1] * n  # girl_match[j] = boy matched to girl j
        
        def bfs_find_augmenting_path():
            """Use BFS to find augmenting path"""
            # Initialize BFS
            queue = deque()
            parent = [-1] * (m + n)  # Parent in BFS tree
            visited = [False] * (m + n)
            
            # Add all unmatched boys to queue
            for boy in range(m):
                if boy_match[boy] == -1:
                    queue.append(boy)
                    visited[boy] = True
            
            augmenting_path_found = False
            
            while queue and not augmenting_path_found:
                current = queue.popleft()
                
                if current < m:  # Current is a boy
                    boy = current
                    for girl in boy_neighbors[boy]:
                        girl_node = m + girl  # Offset for girl nodes
                        
                        if not visited[girl_node]:
                            visited[girl_node] = True
                            parent[girl_node] = boy
                            
                            if girl_match[girl] == -1:
                                # Found augmenting path to unmatched girl
                                augmenting_path_found = True
                                
                                # Trace back and update matching
                                current_girl = girl
                                while current_girl != -1:
                                    matched_boy = parent[m + current_girl]
                                    prev_girl = boy_match[matched_boy] if matched_boy != -1 else -1
                                    
                                    girl_match[current_girl] = matched_boy
                                    boy_match[matched_boy] = current_girl
                                    
                                    current_girl = prev_girl
                                
                                break
                            else:
                                # Girl is matched, add her partner to queue
                                queue.append(girl_match[girl])
                                visited[girl_match[girl]] = True
                                parent[girl_match[girl]] = girl_node
            
            return augmenting_path_found
        
        # Keep finding augmenting paths
        matching_count = 0
        while bfs_find_augmenting_path():
            matching_count += 1
        
        return matching_count
    
    def maximumInvitations_approach3_hopcroft_karp_simulation(self, grid: List[List[int]]) -> int:
        """
        Approach 3: Hopcroft-Karp Algorithm Simulation
        
        Simulate the faster Hopcroft-Karp algorithm for maximum bipartite matching.
        
        Time: O(E * sqrt(V))
        Space: O(V + E)
        """
        m, n = len(grid), len(grid[0])
        
        # Build adjacency list
        boys_adj = [[] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    boys_adj[i].append(j)
        
        # Matching arrays
        pair_boy = [-1] * m
        pair_girl = [-1] * n
        dist = [0] * (m + 1)  # Distance for BFS
        
        def bfs():
            """BFS phase of Hopcroft-Karp"""
            queue = deque()
            
            # Initialize distances
            for boy in range(m):
                if pair_boy[boy] == -1:
                    dist[boy] = 0
                    queue.append(boy)
                else:
                    dist[boy] = float('inf')
            
            dist[m] = float('inf')  # Special NIL vertex
            
            while queue:
                boy = queue.popleft()
                
                if dist[boy] < dist[m]:
                    for girl in boys_adj[boy]:
                        if pair_girl[girl] == -1:
                            if dist[m] == float('inf'):
                                dist[m] = dist[boy] + 1
                        elif dist[pair_girl[girl]] == float('inf'):
                            dist[pair_girl[girl]] = dist[boy] + 1
                            queue.append(pair_girl[girl])
            
            return dist[m] != float('inf')
        
        def dfs(boy):
            """DFS phase of Hopcroft-Karp"""
            if boy != -1:
                for girl in boys_adj[boy]:
                    if pair_girl[girl] == -1 or (dist[pair_girl[girl]] == dist[boy] + 1 and dfs(pair_girl[girl])):
                        pair_girl[girl] = boy
                        pair_boy[boy] = girl
                        return True
                
                dist[boy] = float('inf')
                return False
            
            return True
        
        matching = 0
        
        # Main Hopcroft-Karp loop
        while bfs():
            for boy in range(m):
                if pair_boy[boy] == -1 and dfs(boy):
                    matching += 1
        
        return matching
    
    def maximumInvitations_approach4_network_flow_max_flow(self, grid: List[List[int]]) -> int:
        """
        Approach 4: Network Flow (Max Flow) Solution
        
        Model as maximum flow problem and solve using Ford-Fulkerson.
        
        Time: O(V * E^2) for Ford-Fulkerson
        Space: O(V^2)
        """
        m, n = len(grid), len(grid[0])
        
        # Create flow network
        # Nodes: source(0), boys(1..m), girls(m+1..m+n), sink(m+n+1)
        total_nodes = m + n + 2
        source, sink = 0, total_nodes - 1
        
        # Build capacity matrix
        capacity = [[0] * total_nodes for _ in range(total_nodes)]
        
        # Source to boys (capacity 1)
        for boy in range(1, m + 1):
            capacity[source][boy] = 1
        
        # Girls to sink (capacity 1)
        for girl in range(m + 1, m + n + 1):
            capacity[girl][sink] = 1
        
        # Boys to girls based on grid
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    boy_node = i + 1
                    girl_node = m + 1 + j
                    capacity[boy_node][girl_node] = 1
        
        def bfs_find_path():
            """Find augmenting path using BFS"""
            visited = [False] * total_nodes
            parent = [-1] * total_nodes
            queue = deque([source])
            visited[source] = True
            
            while queue:
                u = queue.popleft()
                
                for v in range(total_nodes):
                    if not visited[v] and capacity[u][v] > 0:
                        visited[v] = True
                        parent[v] = u
                        queue.append(v)
                        
                        if v == sink:
                            return parent
            
            return None
        
        max_flow = 0
        
        # Ford-Fulkerson algorithm
        while True:
            parent = bfs_find_path()
            if parent is None:
                break
            
            # Find minimum capacity along the path
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, capacity[parent[s]][s])
                s = parent[s]
            
            # Update capacities
            v = sink
            while v != source:
                u = parent[v]
                capacity[u][v] -= path_flow
                capacity[v][u] += path_flow
                v = parent[v]
            
            max_flow += path_flow
        
        return max_flow
    
    def maximumInvitations_approach5_greedy_with_optimization(self, grid: List[List[int]]) -> int:
        """
        Approach 5: Greedy Algorithm with Local Optimization
        
        Use greedy approach with local search optimization.
        
        Time: O(m * n * k) where k is optimization iterations
        Space: O(m + n)
        """
        m, n = len(grid), len(grid[0])
        
        def greedy_matching():
            """Initial greedy matching"""
            boy_match = [-1] * m
            girl_match = [-1] * n
            
            # Sort boys by number of potential matches (ascending)
            boys_by_options = []
            for boy in range(m):
                options = sum(grid[boy])
                boys_by_options.append((options, boy))
            
            boys_by_options.sort()
            
            for _, boy in boys_by_options:
                for girl in range(n):
                    if grid[boy][girl] == 1 and girl_match[girl] == -1:
                        boy_match[boy] = girl
                        girl_match[girl] = boy
                        break
            
            return boy_match, girl_match
        
        def local_optimization(boy_match, girl_match):
            """Local optimization using augmenting paths"""
            improved = True
            while improved:
                improved = False
                
                for boy in range(m):
                    if boy_match[boy] == -1:
                        # Try to find augmenting path for unmatched boy
                        visited = set()
                        if find_augmenting_path(boy, boy_match, girl_match, visited):
                            improved = True
                            break
            
            return boy_match, girl_match
        
        def find_augmenting_path(boy, boy_match, girl_match, visited):
            """Find augmenting path using DFS"""
            for girl in range(n):
                if grid[boy][girl] == 1 and girl not in visited:
                    visited.add(girl)
                    
                    if girl_match[girl] == -1 or find_augmenting_path(girl_match[girl], boy_match, girl_match, visited):
                        boy_match[boy] = girl
                        girl_match[girl] = boy
                        return True
            
            return False
        
        # Initial greedy matching
        boy_match, girl_match = greedy_matching()
        
        # Local optimization
        boy_match, girl_match = local_optimization(boy_match, girl_match)
        
        # Count matches
        return sum(1 for match in boy_match if match != -1)
    
    def maximumInvitations_approach6_hungarian_algorithm_adaptation(self, grid: List[List[int]]) -> int:
        """
        Approach 6: Hungarian Algorithm Adaptation
        
        Adapt Hungarian algorithm for unweighted bipartite matching.
        
        Time: O(n^3)
        Space: O(n^2)
        """
        m, n = len(grid), len(grid[0])
        
        # For unweighted maximum bipartite matching, we can use simpler approaches
        # This is a conceptual adaptation showing the connection
        
        def maximum_matching_hungarian_style():
            """Hungarian-style algorithm for maximum matching"""
            
            # Convert to assignment problem
            # Create cost matrix where cost = 0 for valid assignments, infinity otherwise
            max_dim = max(m, n)
            cost_matrix = [[float('inf')] * max_dim for _ in range(max_dim)]
            
            for i in range(m):
                for j in range(n):
                    if grid[i][j] == 1:
                        cost_matrix[i][j] = 0  # No cost for valid assignment
            
            # Use simpler maximum bipartite matching since all valid edges have same weight
            return self.maximumInvitations_approach1_maximum_bipartite_matching_dfs(grid)
        
        return maximum_matching_hungarian_style()

def test_maximum_invitations():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (grid, expected)
        ([[1,1,1],
          [1,0,1],
          [0,0,1]], 3),
        
        ([[1,0,1,0],
          [1,0,0,0],
          [0,0,1,0],
          [0,0,1,1]], 3),
        
        ([[1,0,0,0],
          [0,0,0,0],
          [0,0,0,0]], 1),
        
        ([[1]], 1),
        ([[0]], 0),
        ([[1,1],[1,1]], 2),
        ([[1,0],[0,1]], 2),
        ([[1,1,1],[1,1,1],[1,1,1]], 3),
    ]
    
    approaches = [
        ("DFS Matching", solution.maximumInvitations_approach1_maximum_bipartite_matching_dfs),
        ("BFS Matching", solution.maximumInvitations_approach2_maximum_bipartite_matching_bfs),
        ("Hopcroft-Karp", solution.maximumInvitations_approach3_hopcroft_karp_simulation),
        ("Network Flow", solution.maximumInvitations_approach4_network_flow_max_flow),
        ("Greedy Optimized", solution.maximumInvitations_approach5_greedy_with_optimization),
        ("Hungarian Style", solution.maximumInvitations_approach6_hungarian_algorithm_adaptation),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (grid, expected) in enumerate(test_cases):
            try:
                result = func([row[:] for row in grid])  # Deep copy
                status = "✓" if result == expected else "✗"
                print(f"Test {i+1}: {status} expected={expected}, got={result}")
            except Exception as e:
                print(f"Test {i+1}: ERROR - {str(e)}")

def demonstrate_bipartite_matching_concept():
    """Demonstrate bipartite matching concept"""
    print("\n=== Bipartite Matching Concept Demo ===")
    
    grid = [[1,1,1],
            [1,0,1],
            [0,0,1]]
    
    print(f"Grid representation:")
    for i, row in enumerate(grid):
        print(f"Boy {i}: {row}")
    
    print(f"\nBipartite graph structure:")
    print(f"Boys (left side):  B0, B1, B2")
    print(f"Girls (right side): G0, G1, G2")
    print(f"Edges (who can invite whom):")
    
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                print(f"  B{i} → G{j}")
    
    print(f"\nMatching problem:")
    print(f"  Find maximum set of edges such that:")
    print(f"  • Each boy matched to at most one girl")
    print(f"  • Each girl matched to at most one boy")
    
    print(f"\nOptimal matching (one possible solution):")
    print(f"  B0 → G1")
    print(f"  B1 → G0")
    print(f"  B2 → G2")
    print(f"  Total matches: 3")

def demonstrate_augmenting_path_algorithm():
    """Demonstrate augmenting path algorithm step by step"""
    print("\n=== Augmenting Path Algorithm Demo ===")
    
    grid = [[1,1,0],
            [0,1,1],
            [1,0,1]]
    
    print(f"Grid:")
    for i, row in enumerate(grid):
        print(f"Boy {i}: {row}")
    
    print(f"\nStep-by-step matching process:")
    
    # Initial state
    boy_match = [-1, -1, -1]
    girl_match = [-1, -1, -1]
    
    print(f"\nInitial: No matches")
    print(f"Boy matches: {boy_match}")
    print(f"Girl matches: {girl_match}")
    
    # Try to match Boy 0
    print(f"\nStep 1: Try to match Boy 0")
    print(f"  Boy 0 can invite: Girls 0, 1")
    print(f"  Girl 0 is free → Match B0-G0")
    boy_match[0] = 0
    girl_match[0] = 0
    print(f"  Result: {boy_match}")
    
    # Try to match Boy 1
    print(f"\nStep 2: Try to match Boy 1")
    print(f"  Boy 1 can invite: Girls 1, 2")
    print(f"  Girl 1 is free → Match B1-G1")
    boy_match[1] = 1
    girl_match[1] = 1
    print(f"  Result: {boy_match}")
    
    # Try to match Boy 2
    print(f"\nStep 3: Try to match Boy 2")
    print(f"  Boy 2 can invite: Girls 0, 2")
    print(f"  Girl 0 is taken by Boy 0")
    print(f"  Girl 2 is free → Match B2-G2")
    boy_match[2] = 2
    girl_match[2] = 2
    print(f"  Final result: {boy_match}")
    print(f"  Total matches: 3")

def analyze_algorithm_complexities():
    """Analyze time complexities of different approaches"""
    print("\n=== Algorithm Complexity Analysis ===")
    
    print("Maximum Bipartite Matching Algorithms:")
    
    print("\n1. **DFS-based Augmenting Paths:**")
    print("   • Time: O(V * E) - V augmenting path searches")
    print("   • Space: O(V) - recursion stack")
    print("   • Simple to implement")
    print("   • Good for sparse graphs")
    
    print("\n2. **BFS-based Augmenting Paths:**")
    print("   • Time: O(V * E) - similar to DFS")
    print("   • Space: O(V) - BFS queue")
    print("   • Iterative implementation")
    print("   • Better for memory-constrained environments")
    
    print("\n3. **Hopcroft-Karp Algorithm:**")
    print("   • Time: O(E * sqrt(V)) - optimal for bipartite matching")
    print("   • Space: O(V) - distance arrays")
    print("   • Most efficient for large graphs")
    print("   • Complex implementation")
    
    print("\n4. **Network Flow (Ford-Fulkerson):**")
    print("   • Time: O(V * E^2) - general max flow")
    print("   • Space: O(V^2) - capacity matrix")
    print("   • General framework")
    print("   • Overkill for unweighted matching")
    
    print("\n5. **Greedy with Optimization:**")
    print("   • Time: O(V * E) with local search")
    print("   • Space: O(V) - matching arrays")
    print("   • Good practical performance")
    print("   • Easy to understand and modify")
    
    print("\nSelection Guidelines:")
    print("• **Small graphs (V < 100):** DFS augmenting paths")
    print("• **Large graphs (V > 1000):** Hopcroft-Karp")
    print("• **General flow problems:** Network flow")
    print("• **Quick prototyping:** Greedy approach")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of bipartite matching"""
    print("\n=== Real-World Applications ===")
    
    print("Maximum Bipartite Matching Applications:")
    
    print("\n1. **Job Assignment:**")
    print("   • Workers ↔ Jobs")
    print("   • Each worker has skills for certain jobs")
    print("   • Maximize number of filled positions")
    print("   • Consider preferences and qualifications")
    
    print("\n2. **Course Registration:**")
    print("   • Students ↔ Course Sections")
    print("   • Students have preferences and time constraints")
    print("   • Maximize number of successful registrations")
    print("   • Handle capacity constraints")
    
    print("\n3. **Organ Donation:**")
    print("   • Donors ↔ Recipients")
    print("   • Compatibility based on blood type, genetics")
    print("   • Maximize number of successful transplants")
    print("   • Critical for saving lives")
    
    print("\n4. **Network Resource Allocation:**")
    print("   • Users ↔ Network Resources")
    print("   • Bandwidth, servers, storage allocation")
    print("   • Maximize resource utilization")
    print("   • Dynamic reallocation as demand changes")
    
    print("\n5. **Dating/Social Matching:**")
    print("   • Users ↔ Potential Matches")
    print("   • Based on preferences and compatibility")
    print("   • Maximize number of mutual matches")
    print("   • Consider multiple criteria and weights")

def demonstrate_network_flow_modeling():
    """Demonstrate network flow modeling for bipartite matching"""
    print("\n=== Network Flow Modeling ===")
    
    print("Converting Bipartite Matching to Max Flow:")
    
    print("\n1. **Graph Construction:**")
    print("   • Add source node S")
    print("   • Add sink node T")
    print("   • Boys as intermediate nodes")
    print("   • Girls as intermediate nodes")
    
    print("\n2. **Edge Capacities:**")
    print("   • S → Boy: capacity 1 (each boy can match once)")
    print("   • Girl → T: capacity 1 (each girl can match once)")
    print("   • Boy → Girl: capacity 1 (if invitation possible)")
    
    print("\n3. **Flow Interpretation:**")
    print("   • Flow value = number of matches")
    print("   • Max flow = maximum matching")
    print("   • Integer flows correspond to valid matchings")
    
    print("\n4. **Advantages:**")
    print("   • General framework for extensions")
    print("   • Can handle weighted versions")
    print("   • Supports capacity constraints")
    print("   • Well-studied optimization techniques")
    
    print("\n5. **Extensions:**")
    print("   • Minimum cost maximum flow")
    print("   • Multiple assignment capacities")
    print("   • Priority-based matching")
    print("   • Dynamic flow networks")

def demonstrate_optimization_extensions():
    """Demonstrate optimization extensions and variations"""
    print("\n=== Optimization Extensions ===")
    
    print("Bipartite Matching Variations:")
    
    print("\n1. **Weighted Matching:**")
    print("   • Edges have weights/preferences")
    print("   • Maximize total weight of matching")
    print("   • Hungarian algorithm applies")
    print("   • Applications: preference-based assignment")
    
    print("\n2. **Capacitated Matching:**")
    print("   • Nodes can handle multiple matches")
    print("   • Boys can invite multiple girls (up to capacity)")
    print("   • Girls can accept multiple invitations")
    print("   • Generalized network flow formulation")
    
    print("\n3. **Stable Matching:**")
    print("   • Both sides have preferences")
    print("   • Find matching with no blocking pairs")
    print("   • Gale-Shapley algorithm")
    print("   • Applications: medical residency, school choice")
    
    print("\n4. **Online Matching:**")
    print("   • One side arrives dynamically")
    print("   • Make irrevocable decisions")
    print("   • Competitive ratio analysis")
    print("   • Applications: ad auctions, ride sharing")
    
    print("\n5. **Robust Matching:**")
    print("   • Handle uncertainty in preferences")
    print("   • Worst-case or stochastic optimization")
    print("   • Backup options and contingency planning")
    print("   • Risk-aware decision making")

if __name__ == "__main__":
    test_maximum_invitations()
    demonstrate_bipartite_matching_concept()
    demonstrate_augmenting_path_algorithm()
    analyze_algorithm_complexities()
    demonstrate_real_world_applications()
    demonstrate_network_flow_modeling()
    demonstrate_optimization_extensions()

"""
Maximum Bipartite Matching and Invitation Optimization Concepts:
1. Classical Maximum Bipartite Matching Algorithms
2. Augmenting Path Methods and Network Flow Approaches
3. Algorithm Complexity Analysis and Selection Strategies
4. Real-world Applications in Assignment and Allocation Problems
5. Extensions to Weighted, Capacitated, and Dynamic Matching

Key Problem Insights:
- Classic maximum bipartite matching problem
- Boys and girls form bipartite graph with invitation constraints
- Multiple algorithmic approaches with different complexities
- Foundation for many assignment and allocation problems

Algorithm Strategy:
1. Model as bipartite graph with boys and girls as vertices
2. Find maximum matching using appropriate algorithm
3. Handle edge cases and constraint validation
4. Optimize for problem size and structure

Matching Algorithms:
- DFS augmenting paths: Simple, O(VE) complexity
- BFS augmenting paths: Iterative, similar complexity
- Hopcroft-Karp: Optimal O(E√V) for bipartite graphs
- Network flow: General framework, higher complexity
- Greedy with optimization: Practical, good performance

Optimization Techniques:
- Efficient augmenting path finding
- Early termination and pruning
- Data structure optimization
- Algorithm selection based on graph properties

Real-world Applications:
- Job and task assignment optimization
- Course registration and scheduling
- Resource allocation in distributed systems
- Matching markets and platform design
- Network optimization and load balancing

This problem demonstrates fundamental matching theory
essential for assignment optimization and resource allocation.
"""
