"""
886. Possible Bipartition
Difficulty: Medium (Listed as Easy in syllabus)

Problem:
We want to split a group of n people (labeled from 1 to n) into two groups of any size. 
Each person may dislike some other people, and they should not go into the same group.

Given the integer n and the array dislikes where dislikes[i] = [ai, bi] indicates that 
the person labeled ai does not like the person labeled bi, return true if it is possible 
to split everyone into two groups in this way.

Examples:
Input: n = 4, dislikes = [[1,2],[1,3],[2,4]]
Output: true
Explanation: group1 [1,4], group2 [2,3]

Input: n = 3, dislikes = [[1,2],[1,3],[2,3]]
Output: false
Explanation: At any party, the person labeled 1 must be there. But the person labeled 2 
and the person labeled 3 must also be there, but they hate each other.

Input: n = 5, dislikes = [[1,2],[2,3],[3,4],[4,5],[1,5]]
Output: false

Constraints:
- 1 <= n <= 2000
- 0 <= dislikes.length <= 10^4
- dislikes[i].length == 2
- 1 <= ai, bi <= n
- ai != bi
- All the pairs of dislikes are unique.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque

class Solution:
    def possibleBipartition_approach1_graph_coloring_bfs(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        Approach 1: Graph Coloring with BFS
        
        Build graph from dislikes and use 2-coloring to check bipartiteness.
        
        Time: O(V + E) where V = n, E = len(dislikes)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for a, b in dislikes:
            graph[a].append(b)
            graph[b].append(a)
        
        # Color array: -1 = unvisited, 0 = group1, 1 = group2
        color = [-1] * (n + 1)
        
        # Check each connected component
        for person in range(1, n + 1):
            if color[person] == -1:
                # BFS to color this component
                queue = deque([person])
                color[person] = 0
                
                while queue:
                    current = queue.popleft()
                    current_color = color[current]
                    opposite_color = 1 - current_color
                    
                    for disliked in graph[current]:
                        if color[disliked] == -1:
                            # Assign opposite color
                            color[disliked] = opposite_color
                            queue.append(disliked)
                        elif color[disliked] == current_color:
                            # Conflict: both in same group but dislike each other
                            return False
        
        return True
    
    def possibleBipartition_approach2_dfs_recursive(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        Approach 2: DFS Recursive Coloring
        
        Use recursive DFS to assign groups.
        
        Time: O(V + E)
        Space: O(V + E) for recursion and adjacency list
        """
        # Build graph
        graph = defaultdict(list)
        for a, b in dislikes:
            graph[a].append(b)
            graph[b].append(a)
        
        color = {}
        
        def dfs(person, group):
            """Assign person to group and recursively assign neighbors"""
            if person in color:
                return color[person] == group
            
            color[person] = group
            
            # All disliked people must be in opposite group
            for disliked in graph[person]:
                if not dfs(disliked, 1 - group):
                    return False
            
            return True
        
        # Check each person
        for person in range(1, n + 1):
            if person not in color:
                if not dfs(person, 0):
                    return False
        
        return True
    
    def possibleBipartition_approach3_union_find_approach(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        Approach 3: Union-Find with Enemy Sets
        
        Use Union-Find to track friend and enemy relationships.
        
        Time: O(E * α(V))
        Space: O(V)
        """
        class UnionFind:
            def __init__(self, size):
                self.parent = list(range(size))
                self.rank = [0] * size
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px == py:
                    return
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
            
            def connected(self, x, y):
                return self.find(x) == self.find(y)
        
        # Create UF with 2n nodes: 1..n for group1, n+1..2n for group2
        uf = UnionFind(2 * n + 1)
        
        for a, b in dislikes:
            # Check if a and b are already in same group
            if uf.connected(a, b):
                return False
            
            # Union a with b's enemy group, and b with a's enemy group
            uf.union(a, b + n)  # a is enemy of b's group
            uf.union(b, a + n)  # b is enemy of a's group
        
        return True
    
    def possibleBipartition_approach4_iterative_coloring(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        Approach 4: Iterative Coloring with Stack
        
        Avoid recursion limits with iterative approach.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for a, b in dislikes:
            graph[a].append(b)
            graph[b].append(a)
        
        color = {}
        
        for start in range(1, n + 1):
            if start not in color:
                # Use stack for iterative DFS
                stack = [(start, 0)]
                
                while stack:
                    person, group = stack.pop()
                    
                    if person in color:
                        # Already colored, check consistency
                        if color[person] != group:
                            return False
                        continue
                    
                    color[person] = group
                    
                    # Add all disliked people to opposite group
                    for disliked in graph[person]:
                        stack.append((disliked, 1 - group))
        
        return True
    
    def possibleBipartition_approach5_detailed_analysis(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        Approach 5: Detailed Analysis with Group Tracking
        
        Comprehensive analysis with detailed group assignment tracking.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build graph with detailed tracking
        graph = defaultdict(set)
        for a, b in dislikes:
            graph[a].add(b)
            graph[b].add(a)
        
        # Track group assignments
        group_assignment = {}
        group1 = set()
        group2 = set()
        
        def assign_groups_bfs(start_person):
            """Assign groups using BFS with detailed tracking"""
            queue = deque([(start_person, 1)])  # Start with group 1
            
            while queue:
                person, group = queue.popleft()
                
                if person in group_assignment:
                    if group_assignment[person] != group:
                        return False
                    continue
                
                # Assign to group
                group_assignment[person] = group
                if group == 1:
                    group1.add(person)
                else:
                    group2.add(person)
                
                # Process disliked people
                for disliked in graph[person]:
                    opposite_group = 2 if group == 1 else 1
                    
                    if disliked in group_assignment:
                        if group_assignment[disliked] == group:
                            return False  # Same group conflict
                    else:
                        queue.append((disliked, opposite_group))
            
            return True
        
        # Process each connected component
        for person in range(1, n + 1):
            if person not in group_assignment:
                if not assign_groups_bfs(person):
                    return False
        
        # Verify final assignment
        for person in range(1, n + 1):
            for disliked in graph[person]:
                if group_assignment.get(person) == group_assignment.get(disliked):
                    return False
        
        return True
    
    def possibleBipartition_approach6_conflict_graph_analysis(self, n: int, dislikes: List[List[int]]) -> bool:
        """
        Approach 6: Conflict Graph Analysis
        
        Analyze the conflict graph structure for bipartiteness.
        
        Time: O(V + E)
        Space: O(V + E)
        """
        # Build conflict graph
        conflicts = defaultdict(set)
        for a, b in dislikes:
            conflicts[a].add(b)
            conflicts[b].add(a)
        
        # Analyze graph properties
        visited = set()
        
        def analyze_component(start):
            """Analyze a connected component for bipartiteness"""
            component = set()
            partition1 = set()
            partition2 = set()
            
            queue = deque([(start, 1)])
            
            while queue:
                person, partition = queue.popleft()
                
                if person in component:
                    # Check consistency
                    expected_partition = 1 if person in partition1 else 2
                    if expected_partition != partition:
                        return False
                    continue
                
                component.add(person)
                if partition == 1:
                    partition1.add(person)
                else:
                    partition2.add(person)
                
                # Process conflicts
                for enemy in conflicts[person]:
                    opposite_partition = 2 if partition == 1 else 1
                    queue.append((enemy, opposite_partition))
            
            # Verify no internal conflicts within partitions
            for person in partition1:
                if any(enemy in partition1 for enemy in conflicts[person]):
                    return False
            
            for person in partition2:
                if any(enemy in partition2 for enemy in conflicts[person]):
                    return False
            
            return True
        
        # Check each component
        for person in range(1, n + 1):
            if person not in visited:
                if not analyze_component(person):
                    return False
                
                # Mark component as visited
                queue = deque([person])
                component_visited = set()
                
                while queue:
                    current = queue.popleft()
                    if current in component_visited:
                        continue
                    
                    component_visited.add(current)
                    visited.add(current)
                    
                    for enemy in conflicts[current]:
                        if enemy not in component_visited:
                            queue.append(enemy)
        
        return True

def test_possible_bipartition():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (n, dislikes, expected)
        (4, [[1,2],[1,3],[2,4]], True),
        (3, [[1,2],[1,3],[2,3]], False),
        (5, [[1,2],[2,3],[3,4],[4,5],[1,5]], False),
        (10, [[1,2],[3,4],[5,6],[7,8],[9,10]], True),
        (4, [[1,2],[1,3],[2,3]], False),  # Triangle
        (6, [[1,2],[2,3],[3,1],[4,5],[5,6],[6,4]], False),  # Two triangles
        (2, [[1,2]], True),
        (1, [], True),
        (5, [], True),  # No conflicts
    ]
    
    approaches = [
        ("BFS Coloring", solution.possibleBipartition_approach1_graph_coloring_bfs),
        ("DFS Recursive", solution.possibleBipartition_approach2_dfs_recursive),
        ("Union-Find", solution.possibleBipartition_approach3_union_find_approach),
        ("Iterative Coloring", solution.possibleBipartition_approach4_iterative_coloring),
        ("Detailed Analysis", solution.possibleBipartition_approach5_detailed_analysis),
        ("Conflict Analysis", solution.possibleBipartition_approach6_conflict_graph_analysis),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, dislikes, expected) in enumerate(test_cases):
            result = func(n, [dislike[:] for dislike in dislikes])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, expected={expected}, got={result}")

def demonstrate_bipartition_example():
    """Demonstrate bipartition with detailed example"""
    print("\n=== Bipartition Example Demo ===")
    
    n = 4
    dislikes = [[1,2],[1,3],[2,4]]
    
    print(f"Example: n={n}, dislikes={dislikes}")
    print(f"Conflict relationships:")
    print(f"  Person 1 dislikes: 2, 3")
    print(f"  Person 2 dislikes: 1, 4")
    print(f"  Person 3 dislikes: 1")
    print(f"  Person 4 dislikes: 2")
    
    print(f"\nGraph representation:")
    print(f"  1 -- 2 -- 4")
    print(f"  |")
    print(f"  3")
    
    print(f"\nBipartition process:")
    print(f"  Start with person 1 in Group A")
    print(f"  Person 2 must be in Group B (conflicts with 1)")
    print(f"  Person 3 must be in Group B (conflicts with 1)")
    print(f"  Person 4 must be in Group A (conflicts with 2)")
    
    print(f"\nFinal groups:")
    print(f"  Group A: {{1, 4}}")
    print(f"  Group B: {{2, 3}}")
    print(f"  No conflicts within groups!")
    
    solution = Solution()
    result = solution.possibleBipartition_approach1_graph_coloring_bfs(n, dislikes)
    print(f"\nResult: {result}")

def demonstrate_impossible_case():
    """Demonstrate case where bipartition is impossible"""
    print("\n=== Impossible Bipartition Demo ===")
    
    n = 3
    dislikes = [[1,2],[1,3],[2,3]]
    
    print(f"Example: n={n}, dislikes={dislikes}")
    print(f"Conflict relationships:")
    print(f"  Person 1 dislikes: 2, 3")
    print(f"  Person 2 dislikes: 1, 3")
    print(f"  Person 3 dislikes: 1, 2")
    
    print(f"\nGraph representation (triangle):")
    print(f"  1 -- 2")
    print(f"  |    |")
    print(f"  3 ---+")
    
    print(f"\nWhy impossible:")
    print(f"  If 1 is in Group A:")
    print(f"    → 2 and 3 must be in Group B")
    print(f"    → But 2 and 3 dislike each other!")
    print(f"  Same problem if we start with any person")
    print(f"  Triangle (odd cycle) cannot be 2-colored")
    
    solution = Solution()
    result = solution.possibleBipartition_approach1_graph_coloring_bfs(n, dislikes)
    print(f"\nResult: {result}")

def analyze_conflict_patterns():
    """Analyze different conflict patterns"""
    print("\n=== Conflict Pattern Analysis ===")
    
    print("Common Conflict Patterns:")
    
    print("\n1. **Chain Conflicts:**")
    print("   • Linear chain: 1-2-3-4")
    print("   • Always bipartite")
    print("   • Groups: {1,3} and {2,4}")
    
    print("\n2. **Star Conflicts:**")
    print("   • Central person conflicts with all others")
    print("   • Always bipartite")
    print("   • Groups: {center} and {others}")
    
    print("\n3. **Triangle Conflicts:**")
    print("   • Three mutual conflicts")
    print("   • Never bipartite (odd cycle)")
    print("   • Impossible to separate")
    
    print("\n4. **Bipartite Complete Graph:**")
    print("   • Two groups, all cross-conflicts")
    print("   • Perfectly bipartite by definition")
    print("   • Maximum conflicts while remaining separable")
    
    print("\n5. **Mixed Patterns:**")
    print("   • Multiple components")
    print("   • Some bipartite, some not")
    print("   • Overall bipartiteness depends on all components")

def demonstrate_union_find_approach():
    """Demonstrate Union-Find approach in detail"""
    print("\n=== Union-Find Approach Demo ===")
    
    n = 4
    dislikes = [[1,2],[1,3],[2,4]]
    
    print(f"Example: n={n}, dislikes={dislikes}")
    print(f"Union-Find with enemy groups:")
    print(f"  Nodes 1-4: original people")
    print(f"  Nodes 5-8: enemy groups (person i's enemies in group i+4)")
    
    print(f"\nProcessing conflicts:")
    print(f"  Conflict [1,2]:")
    print(f"    Union(1, 6) - person 1 with person 2's enemy group")
    print(f"    Union(2, 5) - person 2 with person 1's enemy group")
    
    print(f"  Conflict [1,3]:")
    print(f"    Union(1, 7) - person 1 with person 3's enemy group")
    print(f"    Union(3, 5) - person 3 with person 1's enemy group")
    
    print(f"  Conflict [2,4]:")
    print(f"    Union(2, 8) - person 2 with person 4's enemy group")
    print(f"    Union(4, 6) - person 4 with person 2's enemy group")
    
    print(f"\nFinal groups:")
    print(f"  Check if any person is in same component as their enemy group")
    print(f"  If yes → impossible bipartition")
    print(f"  If no → possible bipartition")

def analyze_algorithmic_complexity():
    """Analyze complexity of different approaches"""
    print("\n=== Algorithmic Complexity Analysis ===")
    
    print("Approach Comparison:")
    
    print("\n1. **BFS/DFS Coloring:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Most intuitive approach")
    print("   • Direct graph traversal")
    
    print("\n2. **Union-Find:**")
    print("   • Time: O(E × α(V))")
    print("   • Space: O(V)")
    print("   • Good for dynamic scenarios")
    print("   • Handles incremental updates")
    
    print("\n3. **Iterative Approaches:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V + E)")
    print("   • Avoids recursion depth issues")
    print("   • Better for large graphs")
    
    print("\nPractical Considerations:")
    print("• **Small graphs (n < 100):** Any approach works")
    print("• **Large graphs (n > 1000):** BFS preferred")
    print("• **Dynamic updates:** Union-Find")
    print("• **Memory constrained:** Union-Find")

def demonstrate_real_world_scenarios():
    """Demonstrate real-world applications"""
    print("\n=== Real-World Scenarios ===")
    
    print("Bipartition Applications:")
    
    print("\n1. **Team Formation:**")
    print("   • Split employees into two project teams")
    print("   • Some employees can't work together")
    print("   • Find feasible team assignment")
    
    print("\n2. **Seating Arrangement:**")
    print("   • Two tables at event")
    print("   • Some guests don't get along")
    print("   • Separate problematic pairs")
    
    print("\n3. **Class Scheduling:**")
    print("   • Two time slots available")
    print("   • Some students have conflicts")
    print("   • Minimize scheduling conflicts")
    
    print("\n4. **Resource Allocation:**")
    print("   • Two servers available")
    print("   • Some processes interfere")
    print("   • Assign processes to servers")
    
    print("\n5. **Social Network Analysis:**")
    print("   • Divide users into groups")
    print("   • Avoid putting adversaries together")
    print("   • Maintain community harmony")

if __name__ == "__main__":
    test_possible_bipartition()
    demonstrate_bipartition_example()
    demonstrate_impossible_case()
    analyze_conflict_patterns()
    demonstrate_union_find_approach()
    analyze_algorithmic_complexity()
    demonstrate_real_world_scenarios()

"""
Possible Bipartition and Conflict Resolution Concepts:
1. Graph Coloring and 2-Partitioning Algorithms
2. Conflict Graph Analysis and Resolution Strategies
3. Union-Find Applications in Group Management
4. BFS/DFS Traversal for Constraint Satisfaction
5. Real-world Applications in Team and Resource Management

Key Problem Insights:
- Bipartition problem is equivalent to graph 2-coloring
- Conflicts create edges in an undirected graph
- Graph is bipartite ⟺ no odd cycles ⟺ 2-colorable
- Multiple connected components can be handled independently

Algorithm Strategy:
1. Build conflict graph from dislike relationships
2. Use BFS/DFS to assign people to groups
3. Ensure conflicting people are in different groups
4. Detect impossibility when conflicts create odd cycles

Conflict Resolution Patterns:
- Chain conflicts: always solvable
- Star conflicts: central person vs others
- Triangle conflicts: impossible to resolve
- Complex patterns: component-wise analysis

Optimization Techniques:
- Early termination on conflict detection
- Component-wise processing for efficiency
- Union-Find for dynamic conflict management
- Iterative approaches for large datasets

Real-world Applications:
- Team formation with personality conflicts
- Resource allocation with interference constraints
- Social event planning and seating arrangements
- Process scheduling with mutual exclusions
- Network partitioning with adversarial relationships

This problem demonstrates practical constraint satisfaction
essential for conflict resolution and group management.
"""
