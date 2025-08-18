"""
1466. Reorder Routes to Make All Paths Lead to the City Zero
Difficulty: Medium

Problem:
There are n cities numbered from 0 to n - 1 and n - 1 roads such that there is only one way 
to travel between two different cities (this network form a tree). Last year, The government 
decided that they would select city 0 as the capital of this country.

Given the roads where roads[i] = [ai, bi] indicates that there was a road directed from city ai 
to city bi. Return the minimum number of roads that need to be reversed so that every city can 
reach the capital city 0.

It's guaranteed that every city can reach the capital after reordering.

Examples:
Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
Output: 3

Input: n = 5, connections = [[1,0],[1,2],[3,2],[3,4]]
Output: 2

Input: n = 3, connections = [[1,0],[2,0]]
Output: 0

Constraints:
- 2 <= n <= 5 * 10^4
- connections.length == n - 1
- connections[i].length == 2
- 0 <= ai, bi <= n - 1
- ai != bi
"""

from typing import List
from collections import defaultdict, deque

class Solution:
    def minReorder_approach1_dfs_tree_traversal(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 1: DFS Tree Traversal
        
        Build bidirectional graph but track original directions.
        From city 0, DFS and count edges that need to be reversed.
        
        Time: O(N) - visit each node once
        Space: O(N) - adjacency list and recursion stack
        """
        # Build bidirectional adjacency list with direction info
        graph = defaultdict(list)
        
        for a, b in connections:
            graph[a].append((b, 1))  # Original direction (needs reversal)
            graph[b].append((a, 0))  # Reverse direction (correct)
        
        visited = set()
        reversals = 0
        
        def dfs(city):
            nonlocal reversals
            visited.add(city)
            
            for neighbor, needs_reversal in graph[city]:
                if neighbor not in visited:
                    reversals += needs_reversal
                    dfs(neighbor)
        
        dfs(0)
        return reversals
    
    def minReorder_approach2_bfs_level_order(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 2: BFS Level-order Traversal
        
        Use BFS to traverse tree from root (city 0) and count reversals.
        
        Time: O(N)
        Space: O(N)
        """
        # Build bidirectional graph with direction tracking
        graph = defaultdict(list)
        
        for a, b in connections:
            graph[a].append((b, 1))  # Forward edge (reverse needed)
            graph[b].append((a, 0))  # Backward edge (correct direction)
        
        visited = set([0])
        queue = deque([0])
        reversals = 0
        
        while queue:
            city = queue.popleft()
            
            for neighbor, needs_reversal in graph[city]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    reversals += needs_reversal
                    queue.append(neighbor)
        
        return reversals
    
    def minReorder_approach3_edge_direction_tracking(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 3: Edge Direction Tracking with Sets
        
        Use sets to track original edge directions and traverse tree.
        
        Time: O(N)
        Space: O(N)
        """
        # Build undirected graph
        graph = defaultdict(list)
        # Track original directed edges
        directed_edges = set()
        
        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
            directed_edges.add((a, b))
        
        visited = set()
        reversals = 0
        
        def dfs(city):
            nonlocal reversals
            visited.add(city)
            
            for neighbor in graph[city]:
                if neighbor not in visited:
                    # Check if we're going against original direction
                    if (city, neighbor) in directed_edges:
                        reversals += 1  # Need to reverse this edge
                    dfs(neighbor)
        
        dfs(0)
        return reversals
    
    def minReorder_approach4_iterative_dfs(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 4: Iterative DFS to avoid recursion
        
        Use explicit stack for DFS traversal.
        
        Time: O(N)
        Space: O(N)
        """
        # Build graph with direction info
        graph = defaultdict(list)
        
        for a, b in connections:
            graph[a].append((b, True))   # Forward edge
            graph[b].append((a, False))  # Backward edge
        
        visited = set()
        stack = [0]
        reversals = 0
        
        while stack:
            city = stack.pop()
            
            if city in visited:
                continue
            
            visited.add(city)
            
            for neighbor, is_forward in graph[city]:
                if neighbor not in visited:
                    if is_forward:
                        reversals += 1
                    stack.append(neighbor)
        
        return reversals
    
    def minReorder_approach5_parent_child_relationship(self, n: int, connections: List[List[int]]) -> int:
        """
        Approach 5: Parent-Child Relationship Tracking
        
        Explicitly track parent-child relationships in tree traversal.
        
        Time: O(N)
        Space: O(N)
        """
        # Build undirected graph
        graph = defaultdict(list)
        original_directions = {}
        
        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
            original_directions[(a, b)] = True
            original_directions[(b, a)] = False
        
        visited = set()
        reversals = 0
        
        def dfs(city, parent):
            nonlocal reversals
            visited.add(city)
            
            for neighbor in graph[city]:
                if neighbor != parent and neighbor not in visited:
                    # Check if edge from city to neighbor is in original direction
                    if original_directions.get((city, neighbor), False):
                        reversals += 1
                    dfs(neighbor, city)
        
        dfs(0, -1)
        return reversals

def test_min_reorder():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (n, connections, expected)
        (6, [[0,1],[1,3],[2,3],[4,0],[4,5]], 3),
        (5, [[1,0],[1,2],[3,2],[3,4]], 2),
        (3, [[1,0],[2,0]], 0),
        (4, [[0,1],[0,2],[0,3]], 0),  # All point away from 0
        (4, [[1,0],[2,0],[3,0]], 0),  # All point to 0
        (4, [[0,1],[1,2],[2,3]], 3),  # Linear chain away from 0
        (4, [[3,2],[2,1],[1,0]], 0),  # Linear chain to 0
    ]
    
    approaches = [
        ("DFS Tree Traversal", solution.minReorder_approach1_dfs_tree_traversal),
        ("BFS Level Order", solution.minReorder_approach2_bfs_level_order),
        ("Edge Direction Tracking", solution.minReorder_approach3_edge_direction_tracking),
        ("Iterative DFS", solution.minReorder_approach4_iterative_dfs),
        ("Parent-Child Relationship", solution.minReorder_approach5_parent_child_relationship),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (n, connections, expected) in enumerate(test_cases):
            result = func(n, connections)
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} n={n}, connections={connections}")
            print(f"         Expected: {expected}, Got: {result}")

def demonstrate_route_reordering():
    """Demonstrate the route reordering process"""
    print("\n=== Route Reordering Demo ===")
    
    n = 6
    connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
    
    print(f"Cities: 0 to {n-1}")
    print(f"Original connections: {connections}")
    print(f"Goal: All cities should be able to reach city 0")
    
    # Build visualization
    print(f"\nOriginal directed graph:")
    for a, b in connections:
        print(f"  {a} -> {b}")
    
    # Analyze using DFS from city 0
    graph = defaultdict(list)
    for a, b in connections:
        graph[a].append((b, 1))  # Forward (needs reversal)
        graph[b].append((a, 0))  # Backward (correct)
    
    visited = set()
    reversals = []
    
    def dfs_trace(city, path):
        visited.add(city)
        path.append(city)
        
        for neighbor, needs_reversal in graph[city]:
            if neighbor not in visited:
                if needs_reversal:
                    reversals.append((city, neighbor))
                    print(f"  Need to reverse: {city} -> {neighbor}")
                else:
                    print(f"  Correct direction: {neighbor} -> {city}")
                
                dfs_trace(neighbor, path[:])
    
    print(f"\nTraversing from city 0:")
    dfs_trace(0, [])
    
    print(f"\nEdges that need reversal: {reversals}")
    print(f"Total reversals needed: {len(reversals)}")
    
    print(f"\nAfter reversals, all edges point toward city 0:")
    for a, b in connections:
        if (a, b) in reversals:
            print(f"  {b} -> {a} (reversed)")
        else:
            print(f"  {b} -> {a} (original)")

def visualize_tree_structure():
    """Visualize tree structure and required reversals"""
    print("\n=== Tree Structure Visualization ===")
    
    examples = [
        ("Star from 0", 4, [[0,1],[0,2],[0,3]]),
        ("Star to 0", 4, [[1,0],[2,0],[3,0]]),
        ("Chain from 0", 4, [[0,1],[1,2],[2,3]]),
        ("Chain to 0", 4, [[3,2],[2,1],[1,0]]),
        ("Mixed", 5, [[0,1],[2,1],[1,3],[3,4]]),
    ]
    
    solution = Solution()
    
    for name, n, connections in examples:
        reversals = solution.minReorder_approach1_dfs_tree_traversal(n, connections)
        
        print(f"\n{name} (n={n}):")
        print(f"  Connections: {connections}")
        print(f"  Reversals needed: {reversals}")
        
        # Show tree structure
        graph = defaultdict(list)
        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)
        
        # Build tree representation from root 0
        visited = set()
        tree_structure = {}
        
        def build_tree(node, parent):
            visited.add(node)
            children = []
            for neighbor in graph[node]:
                if neighbor != parent and neighbor not in visited:
                    children.append(neighbor)
                    build_tree(neighbor, node)
            tree_structure[node] = children
        
        build_tree(0, -1)
        
        print(f"  Tree structure from root 0:")
        def print_tree(node, depth=0):
            indent = "  " * (depth + 2)
            print(f"{indent}{node}")
            for child in tree_structure.get(node, []):
                print_tree(child, depth + 1)
        
        print_tree(0)

def analyze_problem_properties():
    """Analyze key properties of the route reordering problem"""
    print("\n=== Problem Properties Analysis ===")
    
    print("Key insights:")
    print("1. Graph forms a tree (n-1 edges, n vertices, connected)")
    print("2. Every city must be reachable from city 0")
    print("3. Only need to reverse minimum edges to achieve this")
    print("4. Tree structure guarantees unique path between any two cities")
    
    print(f"\nAlgorithm approach:")
    print("- Treat as undirected tree rooted at city 0")
    print("- DFS/BFS from root to visit all cities")
    print("- Count edges going away from root (these need reversal)")
    print("- Edges pointing toward root are already correct")
    
    print(f"\nComplexity analysis:")
    print("- Time: O(N) - visit each city once")
    print("- Space: O(N) - adjacency list and recursion/queue")
    print("- Optimal: Must examine each edge to determine direction")
    
    print(f"\nReal-world applications:")
    applications = [
        "Traffic flow optimization (all roads lead to capital)",
        "Network routing (all nodes route to central server)",
        "Supply chain (all locations send to distribution center)",
        "Communication networks (all nodes report to headquarters)",
        "Organizational hierarchy (all report to CEO)",
    ]
    
    for app in applications:
        print(f"- {app}")

if __name__ == "__main__":
    test_min_reorder()
    demonstrate_route_reordering()
    visualize_tree_structure()
    analyze_problem_properties()

"""
Graph Theory Concepts:
1. Tree Traversal with Direction Awareness
2. Root-oriented Tree Problems
3. Edge Direction Analysis
4. Minimum Edge Reversal

Key Problem Concepts:
- Tree structure: n-1 edges connecting n cities
- Root at city 0: All paths should lead to the capital
- Edge reversal: Change direction of roads to achieve goal
- Minimum reversals: Find optimal solution

Algorithm Strategy:
- Treat as undirected tree rooted at city 0
- Traverse tree (DFS/BFS) from root
- Track original edge directions
- Count edges pointing away from root (need reversal)
- Edges pointing toward root are already correct

Tree Traversal Techniques:
- DFS with direction tracking
- BFS level-by-level processing
- Parent-child relationship maintenance
- Iterative approaches for large inputs

Real-world Applications:
- Transportation network optimization
- Communication routing protocols
- Organizational structure design
- Supply chain logistics
- Data flow in distributed systems

This problem beautifully combines tree traversal with practical
optimization, showing how graph algorithms solve real infrastructure problems.
"""
