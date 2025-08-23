"""
1059. All Paths from Source Lead to Destination - Multiple Approaches
Difficulty: Easy

Given the edges of a directed graph where edges[i] = [ai, bi] indicates there is an edge between nodes ai and bi, and two nodes source and destination of this graph, determine whether or not all paths starting from source eventually end at destination, that is:

At least one path exists from the source node to the destination node.
If a path exists from the source node to a node with no outgoing edges, then that node must be equal to destination.
The number of possible paths from source to destination is a finite number.

Return true if and only if all roads from source lead to destination.
"""

from typing import List, Set
from collections import defaultdict

class AllPathsLeadToDestination:
    """Multiple approaches to check if all paths lead to destination"""
    
    def leadsToDestination_dfs_cycle_detection(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 1: DFS with Cycle Detection
        
        Use DFS to check all paths and detect cycles.
        
        Time: O(V + E), Space: O(V)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Check if destination has outgoing edges (invalid)
        if graph[destination]:
            return False
        
        # DFS with cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        
        def dfs(node: int) -> bool:
            if color[node] == GRAY:  # Cycle detected
                return False
            if color[node] == BLACK:  # Already processed
                return True
            
            # If node has no outgoing edges, it must be destination
            if not graph[node]:
                return node == destination
            
            color[node] = GRAY
            
            # Check all neighbors
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    return False
            
            color[node] = BLACK
            return True
        
        return dfs(source)
    
    def leadsToDestination_iterative_dfs(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 2: Iterative DFS with Stack
        
        Use iterative DFS to avoid recursion depth issues.
        
        Time: O(V + E), Space: O(V)
        """
        # Build graph
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Destination cannot have outgoing edges
        if graph[destination]:
            return False
        
        # Iterative DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n
        stack = [source]
        
        while stack:
            node = stack[-1]
            
            if color[node] == WHITE:
                color[node] = GRAY
                
                # If leaf node, must be destination
                if not graph[node]:
                    if node != destination:
                        return False
                    color[node] = BLACK
                    stack.pop()
                    continue
                
                # Add neighbors to stack
                for neighbor in graph[node]:
                    if color[neighbor] == GRAY:  # Cycle detected
                        return False
                    if color[neighbor] == WHITE:
                        stack.append(neighbor)
            
            elif color[node] == GRAY:
                # All neighbors processed, mark as black
                color[node] = BLACK
                stack.pop()
            
            else:  # BLACK
                stack.pop()
        
        return True
    
    def leadsToDestination_memoized_dfs(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 3: DFS with Memoization
        
        Use memoization to avoid recomputing results for visited nodes.
        
        Time: O(V + E), Space: O(V)
        """
        # Build graph
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Destination cannot have outgoing edges
        if graph[destination]:
            return False
        
        # Memoization
        memo = {}
        visiting = set()
        
        def dfs(node: int) -> bool:
            if node in memo:
                return memo[node]
            
            if node in visiting:  # Cycle detected
                return False
            
            # Leaf node must be destination
            if not graph[node]:
                memo[node] = (node == destination)
                return memo[node]
            
            visiting.add(node)
            
            # Check all paths from this node
            result = True
            for neighbor in graph[node]:
                if not dfs(neighbor):
                    result = False
                    break
            
            visiting.remove(node)
            memo[node] = result
            return result
        
        return dfs(source)
    
    def leadsToDestination_topological_validation(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 4: Topological Sort Validation
        
        Use topological concepts to validate the graph structure.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Build graph and reverse graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        in_degree = [0] * n
        
        for u, v in edges:
            graph[u].append(v)
            reverse_graph[v].append(u)
            in_degree[v] += 1
        
        # Destination cannot have outgoing edges
        if graph[destination]:
            return False
        
        # Check if destination is reachable from source
        def is_reachable(start: int, target: int) -> bool:
            visited = set()
            stack = [start]
            
            while stack:
                node = stack.pop()
                if node == target:
                    return True
                
                if node not in visited:
                    visited.add(node)
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            return False
        
        if not is_reachable(source, destination):
            return False
        
        # Check if all leaf nodes reachable from source are destination
        def find_all_reachable_leaves(start: int) -> Set[int]:
            visited = set()
            leaves = set()
            stack = [start]
            
            while stack:
                node = stack.pop()
                
                if node not in visited:
                    visited.add(node)
                    
                    if not graph[node]:  # Leaf node
                        leaves.add(node)
                    else:
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                stack.append(neighbor)
            
            return leaves
        
        reachable_leaves = find_all_reachable_leaves(source)
        return len(reachable_leaves) == 1 and destination in reachable_leaves
    
    def leadsToDestination_path_analysis(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        Approach 5: Comprehensive Path Analysis
        
        Analyze all possible paths systematically.
        
        Time: O(V + E), Space: O(V)
        """
        # Build adjacency list
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Destination must be a sink (no outgoing edges)
        if graph[destination]:
            return False
        
        # Track node states during DFS
        UNVISITED, VISITING, VISITED = 0, 1, 2
        state = [UNVISITED] * n
        
        def analyze_paths(node: int) -> bool:
            if state[node] == VISITING:  # Cycle detected
                return False
            if state[node] == VISITED:  # Already analyzed
                return True
            
            state[node] = VISITING
            
            # If no outgoing edges, must be destination
            if not graph[node]:
                state[node] = VISITED
                return node == destination
            
            # Analyze all outgoing paths
            for neighbor in graph[node]:
                if not analyze_paths(neighbor):
                    return False
            
            state[node] = VISITED
            return True
        
        return analyze_paths(source)

def test_all_paths_lead_to_destination():
    """Test all paths lead to destination algorithms"""
    solver = AllPathsLeadToDestination()
    
    test_cases = [
        (3, [[0,1],[0,2]], 0, 2, False, "Multiple destinations"),
        (4, [[0,1],[0,3],[1,2],[2,1]], 0, 3, False, "Cycle exists"),
        (4, [[0,1],[0,2],[1,3],[2,3]], 0, 3, True, "All paths lead to destination"),
        (2, [[0,1],[1,1]], 0, 1, False, "Self loop at destination"),
        (3, [[0,1],[1,2]], 0, 2, True, "Simple linear path"),
    ]
    
    algorithms = [
        ("DFS Cycle Detection", solver.leadsToDestination_dfs_cycle_detection),
        ("Iterative DFS", solver.leadsToDestination_iterative_dfs),
        ("Memoized DFS", solver.leadsToDestination_memoized_dfs),
        ("Topological Validation", solver.leadsToDestination_topological_validation),
        ("Path Analysis", solver.leadsToDestination_path_analysis),
    ]
    
    print("=== Testing All Paths Lead to Destination ===")
    
    for n, edges, source, destination, expected, description in test_cases:
        print(f"\n--- {description} (Expected: {expected}) ---")
        print(f"n={n}, edges={edges}, source={source}, dest={destination}")
        
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func(n, edges, source, destination)
                status = "✓" if result == expected else "✗"
                print(f"{alg_name:22} | {status} | Result: {result}")
            except Exception as e:
                print(f"{alg_name:22} | ERROR: {str(e)[:30]}")

if __name__ == "__main__":
    test_all_paths_lead_to_destination()

"""
All Paths Lead to Destination demonstrates path analysis
and cycle detection in directed graphs with destination
validation and comprehensive graph traversal techniques.
"""
