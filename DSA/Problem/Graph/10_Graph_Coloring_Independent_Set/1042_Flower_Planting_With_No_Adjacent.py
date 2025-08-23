"""
1042. Flower Planting With No Adjacent
Difficulty: Easy

Problem:
You have n gardens, numbered from 1 to n, and an array paths where paths[i] = [xi, yi] 
describes a bidirectional path between garden xi to garden yi. In each garden, you want 
to plant one of 4 types of flowers.

All gardens have at most 3 neighbors.

There is no garden that connects to more than 3 other gardens.

Return any such a choice as an array answer, where answer[i] is the type of flower 
planted in the (i+1)-th garden. The flower types are denoted 1, 2, 3, or 4. 
It is guaranteed that an answer exists.

Examples:
Input: n = 3, paths = [[1,2],[2,3],[3,1]]
Output: [1,2,3]
Explanation: 
Gardens 1 and 3 are connected, as well as gardens 2 and 3. Since garden 3 has connections to both other gardens, it cannot have the same type of flower as either of them.
We can choose 1 for garden 3, and 2 for gardens 1 and 2.

Input: n = 4, paths = [[1,2],[3,4]]
Output: [1,2,1,2]

Input: n = 4, paths = [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]
Output: [1,2,3,4]

Constraints:
- 1 <= n <= 10^4
- 0 <= paths.length <= 2 * 10^4
- paths[i].length == 2
- 1 <= xi, yi <= n
- xi != yi
- Every garden has at most 3 neighbors.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

class Solution:
    def gardenNoAdj_approach1_greedy_sequential(self, n: int, paths: List[List[int]]) -> List[int]:
        """
        Approach 1: Greedy Sequential Coloring
        
        Process gardens in order and assign first available flower type.
        
        Time: O(n + m) where m = len(paths)
        Space: O(n + m)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in paths:
            graph[u].append(v)
            graph[v].append(u)
        
        # Initialize flower assignments
        flowers = [0] * (n + 1)  # 1-indexed, flowers[0] unused
        
        # Process each garden
        for garden in range(1, n + 1):
            # Find flowers used by neighbors
            used_flowers = set()
            for neighbor in graph[garden]:
                if flowers[neighbor] != 0:
                    used_flowers.add(flowers[neighbor])
            
            # Find first available flower (1, 2, 3, or 4)
            for flower_type in range(1, 5):
                if flower_type not in used_flowers:
                    flowers[garden] = flower_type
                    break
        
        return flowers[1:]  # Return 1-indexed to 0-indexed
    
    def gardenNoAdj_approach2_degree_ordering(self, n: int, paths: List[List[int]]) -> List[int]:
        """
        Approach 2: Degree-Based Ordering
        
        Process gardens in order of decreasing degree for better coloring.
        
        Time: O(n log n + m)
        Space: O(n + m)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in paths:
            graph[u].append(v)
            graph[v].append(u)
        
        # Sort gardens by degree (highest first)
        gardens_by_degree = sorted(range(1, n + 1), key=lambda x: len(graph[x]), reverse=True)
        
        flowers = [0] * (n + 1)
        
        # Process gardens in degree order
        for garden in gardens_by_degree:
            # Find flowers used by neighbors
            used_flowers = set()
            for neighbor in graph[garden]:
                if flowers[neighbor] != 0:
                    used_flowers.add(flowers[neighbor])
            
            # Assign first available flower
            for flower_type in range(1, 5):
                if flower_type not in used_flowers:
                    flowers[garden] = flower_type
                    break
        
        return flowers[1:]
    
    def gardenNoAdj_approach3_dfs_coloring(self, n: int, paths: List[List[int]]) -> List[int]:
        """
        Approach 3: DFS-based Coloring
        
        Use DFS to traverse and color connected components.
        
        Time: O(n + m)
        Space: O(n + m)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in paths:
            graph[u].append(v)
            graph[v].append(u)
        
        flowers = [0] * (n + 1)
        visited = [False] * (n + 1)
        
        def dfs(garden):
            """DFS to color connected component"""
            visited[garden] = True
            
            # Find flowers used by neighbors
            used_flowers = set()
            for neighbor in graph[garden]:
                if flowers[neighbor] != 0:
                    used_flowers.add(flowers[neighbor])
            
            # Assign first available flower
            for flower_type in range(1, 5):
                if flower_type not in used_flowers:
                    flowers[garden] = flower_type
                    break
            
            # Continue DFS to unvisited neighbors
            for neighbor in graph[garden]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        # Process each connected component
        for garden in range(1, n + 1):
            if not visited[garden]:
                dfs(garden)
        
        return flowers[1:]
    
    def gardenNoAdj_approach4_bfs_coloring(self, n: int, paths: List[List[int]]) -> List[int]:
        """
        Approach 4: BFS-based Coloring
        
        Use BFS level-by-level coloring for systematic approach.
        
        Time: O(n + m)
        Space: O(n + m)
        """
        from collections import deque
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in paths:
            graph[u].append(v)
            graph[v].append(u)
        
        flowers = [0] * (n + 1)
        visited = [False] * (n + 1)
        
        def bfs(start):
            """BFS to color connected component"""
            queue = deque([start])
            visited[start] = True
            
            while queue:
                garden = queue.popleft()
                
                # Find flowers used by neighbors
                used_flowers = set()
                for neighbor in graph[garden]:
                    if flowers[neighbor] != 0:
                        used_flowers.add(flowers[neighbor])
                
                # Assign first available flower if not already assigned
                if flowers[garden] == 0:
                    for flower_type in range(1, 5):
                        if flower_type not in used_flowers:
                            flowers[garden] = flower_type
                            break
                
                # Add unvisited neighbors to queue
                for neighbor in graph[garden]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        
        # Process each connected component
        for garden in range(1, n + 1):
            if not visited[garden]:
                bfs(garden)
        
        return flowers[1:]
    
    def gardenNoAdj_approach5_optimal_coloring(self, n: int, paths: List[List[int]]) -> List[int]:
        """
        Approach 5: Optimal Coloring with Validation
        
        Enhanced approach with validation and optimization.
        
        Time: O(n + m)
        Space: O(n + m)
        """
        # Build adjacency list
        graph = defaultdict(set)  # Use set for O(1) lookup
        for u, v in paths:
            graph[u].add(v)
            graph[v].add(u)
        
        flowers = [0] * (n + 1)
        
        # Process gardens with sophisticated strategy
        def color_garden(garden):
            """Color a single garden optimally"""
            # Find flowers used by already-colored neighbors
            used_flowers = {flowers[neighbor] for neighbor in graph[garden] if flowers[neighbor] != 0}
            
            # Find first available flower
            for flower_type in range(1, 5):
                if flower_type not in used_flowers:
                    return flower_type
            
            # This should never happen given problem constraints
            raise ValueError(f"Cannot color garden {garden}")
        
        # Color gardens in order (greedy is sufficient given constraints)
        for garden in range(1, n + 1):
            flowers[garden] = color_garden(garden)
        
        # Validate solution
        for u, v in paths:
            if flowers[u] == flowers[v]:
                raise ValueError(f"Invalid coloring: gardens {u} and {v} have same flower {flowers[u]}")
        
        return flowers[1:]
    
    def gardenNoAdj_approach6_component_analysis(self, n: int, paths: List[List[int]]) -> List[int]:
        """
        Approach 6: Connected Component Analysis
        
        Analyze each connected component separately for optimal coloring.
        
        Time: O(n + m)
        Space: O(n + m)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in paths:
            graph[u].append(v)
            graph[v].append(u)
        
        flowers = [0] * (n + 1)
        visited = [False] * (n + 1)
        
        def find_component(start):
            """Find all gardens in the connected component"""
            component = []
            stack = [start]
            
            while stack:
                garden = stack.pop()
                if visited[garden]:
                    continue
                
                visited[garden] = True
                component.append(garden)
                
                for neighbor in graph[garden]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            
            return component
        
        def color_component(component):
            """Color gardens in a connected component"""
            # Sort by degree within component for better coloring
            component.sort(key=lambda x: len(graph[x]), reverse=True)
            
            for garden in component:
                # Find flowers used by neighbors
                used_flowers = set()
                for neighbor in graph[garden]:
                    if flowers[neighbor] != 0:
                        used_flowers.add(flowers[neighbor])
                
                # Assign first available flower
                for flower_type in range(1, 5):
                    if flower_type not in used_flowers:
                        flowers[garden] = flower_type
                        break
        
        # Process each connected component
        for garden in range(1, n + 1):
            if not visited[garden]:
                component = find_component(garden)
                # Reset visited for this component
                for g in component:
                    visited[g] = False
                color_component(component)
                # Mark as visited
                for g in component:
                    visited[g] = True
        
        return flowers[1:]

def test_flower_planting():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (n, paths, description)
        (3, [[1,2],[2,3],[3,1]], "Triangle"),
        (4, [[1,2],[3,4]], "Two disconnected edges"),
        (4, [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]], "Complete graph K4"),
        (5, [[1,2],[2,3],[3,4],[4,5]], "Path graph"),
        (6, [[1,2],[1,3],[1,4]], "Star graph"),
        (1, [], "Single garden"),
        (4, [[1,2]], "Single edge"),
    ]
    
    approaches = [
        ("Greedy Sequential", solution.gardenNoAdj_approach1_greedy_sequential),
        ("Degree Ordering", solution.gardenNoAdj_approach2_degree_ordering),
        ("DFS Coloring", solution.gardenNoAdj_approach3_dfs_coloring),
        ("BFS Coloring", solution.gardenNoAdj_approach4_bfs_coloring),
        ("Optimal Coloring", solution.gardenNoAdj_approach5_optimal_coloring),
        ("Component Analysis", solution.gardenNoAdj_approach6_component_analysis),
    ]
    
    for i, (n, paths, desc) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {desc} (n={n}) ---")
        print(f"Paths: {paths}")
        
        for approach_name, func in approaches:
            try:
                result = func(n, paths[:])  # Copy paths
                
                # Validate result
                valid = validate_flower_assignment(n, paths, result)
                colors_used = len(set(result))
                
                status = "✓" if valid else "✗"
                print(f"{approach_name:18} | {status} | Result: {result} | Colors: {colors_used}")
                
            except Exception as e:
                print(f"{approach_name:18} | ERROR: {str(e)}")

def validate_flower_assignment(n: int, paths: List[List[int]], flowers: List[int]) -> bool:
    """Validate that flower assignment is correct"""
    if len(flowers) != n:
        return False
    
    # Check all flowers are in range [1, 4]
    if not all(1 <= f <= 4 for f in flowers):
        return False
    
    # Check no adjacent gardens have same flower
    for u, v in paths:
        if flowers[u-1] == flowers[v-1]:  # Convert to 0-indexed
            return False
    
    return True

def demonstrate_coloring_strategy():
    """Demonstrate flower planting strategy"""
    print("\n=== Flower Planting Strategy Demo ===")
    
    n = 4
    paths = [[1,2],[2,3],[3,4],[4,1]]  # 4-cycle
    
    print(f"Gardens: {list(range(1, n+1))}")
    print(f"Paths: {paths}")
    print(f"Graph structure: 1-2-3-4-1 (square)")
    
    solution = Solution()
    result = solution.gardenNoAdj_approach1_greedy_sequential(n, paths)
    
    print(f"\nGreedy coloring process:")
    
    # Build adjacency list for demo
    graph = defaultdict(list)
    for u, v in paths:
        graph[u].append(v)
        graph[v].append(u)
    
    print(f"Garden 1: neighbors={graph[1]}, available=[1,2,3,4], chosen={result[0]}")
    print(f"Garden 2: neighbors={graph[2]}, neighbor_flowers=[{result[0]}], chosen={result[1]}")
    print(f"Garden 3: neighbors={graph[3]}, neighbor_flowers=[{result[1]}], chosen={result[2]}")
    print(f"Garden 4: neighbors={graph[4]}, neighbor_flowers=[{result[2]}, {result[0]}], chosen={result[3]}")
    
    print(f"\nFinal assignment: {result}")
    print(f"Validation: {validate_flower_assignment(n, paths, result)}")

def demonstrate_constraint_analysis():
    """Demonstrate constraint analysis for flower planting"""
    print("\n=== Constraint Analysis Demo ===")
    
    print("Problem Constraints:")
    print("1. Each garden has at most 3 neighbors")
    print("2. 4 flower types available: [1, 2, 3, 4]")
    print("3. Adjacent gardens must have different flowers")
    
    print(f"\nWhy solution always exists:")
    print(f"• Maximum degree Δ = 3")
    print(f"• Available colors k = 4")
    print(f"• Since k > Δ, greedy coloring always works")
    print(f"• Each garden can always find an available flower")
    
    print(f"\nWorst case scenario:")
    print(f"• Garden with 3 neighbors using flowers [1, 2, 3]")
    print(f"• Current garden can use flower 4")
    print(f"• Always at least one flower available")
    
    # Demonstrate worst case
    n = 4
    paths = [[1,2],[1,3],[1,4]]  # Star with center having 3 neighbors
    
    print(f"\nWorst case example:")
    print(f"n={n}, paths={paths}")
    print(f"Garden 1 has 3 neighbors: [2, 3, 4]")
    
    solution = Solution()
    result = solution.gardenNoAdj_approach1_greedy_sequential(n, paths)
    
    print(f"Solution: {result}")
    print(f"Garden 1 (center): flower {result[0]}")
    print(f"Garden 2: flower {result[1]}")
    print(f"Garden 3: flower {result[2]}")
    print(f"Garden 4: flower {result[3]}")

def analyze_algorithm_efficiency():
    """Analyze efficiency of different approaches"""
    print("\n=== Algorithm Efficiency Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **Greedy Sequential:**")
    print("   • Time: O(n + m) - optimal")
    print("   • Space: O(n + m)")
    print("   • Simple and efficient")
    print("   • Processes gardens in order 1..n")
    
    print("\n2. **Degree Ordering:**")
    print("   • Time: O(n log n + m) - sorting overhead")
    print("   • Space: O(n + m)")
    print("   • Better for general graphs")
    print("   • Overkill for this problem")
    
    print("\n3. **DFS/BFS Coloring:**")
    print("   • Time: O(n + m)")
    print("   • Space: O(n + m)")
    print("   • Handles disconnected components well")
    print("   • More complex than needed")
    
    print("\n4. **Component Analysis:**")
    print("   • Time: O(n + m)")
    print("   • Space: O(n + m)")
    print("   • Most sophisticated approach")
    print("   • Good for understanding graph structure")
    
    print("\nRecommendation:")
    print("• **Use Greedy Sequential** for this problem")
    print("• Simple, optimal time complexity")
    print("• Guaranteed to work given constraints")
    print("• Easy to implement and understand")

def demonstrate_graph_coloring_connection():
    """Demonstrate connection to general graph coloring"""
    print("\n=== Graph Coloring Connection ===")
    
    print("Flower Planting as Graph Coloring:")
    
    print("\n1. **Problem Mapping:**")
    print("   • Gardens → Vertices")
    print("   • Paths → Edges")
    print("   • Flower types → Colors")
    print("   • Adjacent constraint → Proper coloring")
    
    print("\n2. **Special Properties:**")
    print("   • Maximum degree Δ = 3")
    print("   • Available colors k = 4")
    print("   • k > Δ guarantees solution exists")
    print("   • Greedy coloring is optimal")
    
    print("\n3. **General Graph Coloring vs This Problem:**")
    print("   • General: Find χ(G) (chromatic number)")
    print("   • This problem: Color with 4 colors (always possible)")
    print("   • General: NP-complete")
    print("   • This problem: Polynomial time")
    
    print("\n4. **Theoretical Guarantees:**")
    print("   • Brooks' theorem: χ(G) ≤ Δ(G) for connected non-complete graphs")
    print("   • Here: χ(G) ≤ 3 for connected components")
    print("   • With 4 colors available, always sufficient")
    
    print("\n5. **Algorithm Selection:**")
    print("   • Any greedy algorithm works")
    print("   • No need for sophisticated heuristics")
    print("   • Linear time complexity achievable")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications similar to flower planting"""
    print("\n=== Real-World Applications ===")
    
    print("Problems Similar to Flower Planting:")
    
    print("\n1. **Frequency Assignment:**")
    print("   • Radio stations → Gardens")
    print("   • Interference → Adjacency")
    print("   • Frequencies → Flower types")
    print("   • Avoid interference between nearby stations")
    
    print("\n2. **Course Scheduling:**")
    print("   • Courses → Gardens")
    print("   • Student conflicts → Adjacency")
    print("   • Time slots → Flower types")
    print("   • Students can't take conflicting courses simultaneously")
    
    print("\n3. **Register Allocation:**")
    print("   • Variables → Gardens")
    print("   • Live ranges → Adjacency")
    print("   • CPU registers → Flower types")
    print("   • Variables with overlapping lifetimes need different registers")
    
    print("\n4. **Map Coloring:**")
    print("   • Regions → Gardens")
    print("   • Borders → Adjacency")
    print("   • Colors → Flower types")
    print("   • Adjacent regions need different colors")
    
    print("\n5. **Resource Allocation:**")
    print("   • Tasks → Gardens")
    print("   • Conflicts → Adjacency")
    print("   • Resources → Flower types")
    print("   • Conflicting tasks need different resources")

if __name__ == "__main__":
    test_flower_planting()
    demonstrate_coloring_strategy()
    demonstrate_constraint_analysis()
    analyze_algorithm_efficiency()
    demonstrate_graph_coloring_connection()
    demonstrate_real_world_applications()

"""
Flower Planting and Constraint Graph Coloring Concepts:
1. Simple Graph Coloring with Degree Constraints
2. Greedy Coloring Algorithms and Optimality Conditions
3. Connected Component Analysis and Processing
4. Constraint Satisfaction with Guaranteed Solutions
5. Real-world Applications in Resource Allocation

Key Problem Insights:
- Special case of graph coloring with guaranteed solution
- Maximum degree constraint (≤3) with 4 colors available
- Greedy algorithms are optimal for this problem
- Linear time complexity achievable
- Demonstrates fundamental graph coloring principles

Algorithm Strategy:
1. Build adjacency list from paths
2. Process gardens in systematic order
3. For each garden, find first available flower type
4. Assign flower avoiding neighbor conflicts

Theoretical Foundation:
- Brooks' theorem: χ(G) ≤ Δ(G) for non-complete graphs
- With Δ = 3 and 4 colors available, solution always exists
- Greedy coloring sufficient for optimality
- No need for sophisticated heuristics

Optimization Techniques:
- Simple greedy sequential processing
- Degree-based ordering for general improvement
- Connected component analysis for structure
- Efficient data structures for neighbor lookup

Real-world Applications:
- Frequency assignment in telecommunications
- Course and exam scheduling
- Register allocation in compilers
- Map coloring and visualization
- Resource allocation with conflicts

This problem demonstrates how constraint satisfaction
becomes tractable with favorable problem structure.
"""
