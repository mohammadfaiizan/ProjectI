"""
Hamiltonian Path Detection - Multiple Approaches
Difficulty: Medium

A Hamiltonian path is a path that visits each vertex exactly once.
A Hamiltonian cycle is a Hamiltonian path that returns to the starting vertex.

Key Concepts:
1. Backtracking Algorithm
2. Dynamic Programming with Bitmask
3. Graph Reduction Techniques
4. Heuristic Approaches
5. Special Case Optimizations
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import itertools

class HamiltonianPathDetection:
    """Multiple approaches for Hamiltonian path detection"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.adjacency_matrix = None
    
    def add_edge(self, u: int, v: int, directed: bool = False):
        """Add edge to graph"""
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)
        
        if not directed:
            self.graph[v].append(u)
    
    def build_adjacency_matrix(self):
        """Build adjacency matrix for efficient lookup"""
        n = len(self.vertices)
        vertex_to_idx = {v: i for i, v in enumerate(sorted(self.vertices))}
        self.vertex_to_idx = vertex_to_idx
        self.idx_to_vertex = {i: v for v, i in vertex_to_idx.items()}
        
        self.adjacency_matrix = [[False] * n for _ in range(n)]
        
        for u in self.graph:
            for v in self.graph[u]:
                i, j = vertex_to_idx[u], vertex_to_idx[v]
                self.adjacency_matrix[i][j] = True
    
    def has_hamiltonian_path_backtrack(self) -> bool:
        """
        Approach 1: Backtracking Algorithm
        
        Try all possible paths using backtracking.
        
        Time: O(n!), Space: O(n)
        """
        if not self.vertices:
            return True
        
        n = len(self.vertices)
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()
        
        def backtrack(path: List[int], visited: Set[int]) -> bool:
            if len(path) == n:
                return True
            
            current = path[-1] if path else -1
            
            for next_vertex in range(n):
                if next_vertex in visited:
                    continue
                
                # Check if there's an edge (or if starting)
                if current == -1 or self.adjacency_matrix[current][next_vertex]:
                    path.append(next_vertex)
                    visited.add(next_vertex)
                    
                    if backtrack(path, visited):
                        return True
                    
                    path.pop()
                    visited.remove(next_vertex)
            
            return False
        
        # Try starting from each vertex
        for start in range(n):
            if backtrack([start], {start}):
                return True
        
        return False
    
    def has_hamiltonian_path_dp_bitmask(self) -> bool:
        """
        Approach 2: Dynamic Programming with Bitmask
        
        Use DP where dp[mask][i] = True if we can visit nodes in mask ending at i.
        
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        if not self.vertices:
            return True
        
        n = len(self.vertices)
        if n == 1:
            return True
        
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()
        
        # dp[mask][i] = True if we can visit nodes in mask, ending at node i
        dp = [[False] * n for _ in range(1 << n)]
        
        # Initialize: single nodes
        for i in range(n):
            dp[1 << i][i] = True
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if not dp[mask][u]:
                    continue
                
                # Try to extend to unvisited neighbors
                for v in range(n):
                    if (mask & (1 << v)) == 0 and self.adjacency_matrix[u][v]:
                        new_mask = mask | (1 << v)
                        dp[new_mask][v] = True
        
        # Check if any configuration visits all nodes
        full_mask = (1 << n) - 1
        return any(dp[full_mask][i] for i in range(n))
    
    def has_hamiltonian_cycle_dp_bitmask(self) -> bool:
        """
        Approach 3: Hamiltonian Cycle Detection using DP
        
        Similar to path but must return to starting vertex.
        
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        if not self.vertices:
            return True
        
        n = len(self.vertices)
        if n < 3:
            return False
        
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()
        
        # dp[mask][i] = True if we can visit nodes in mask, ending at node i
        dp = [[False] * n for _ in range(1 << n)]
        
        # Start from vertex 0
        dp[1][0] = True
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if not dp[mask][u]:
                    continue
                
                for v in range(n):
                    if (mask & (1 << v)) == 0 and self.adjacency_matrix[u][v]:
                        new_mask = mask | (1 << v)
                        dp[new_mask][v] = True
        
        # Check if we can return to start from any ending vertex
        full_mask = (1 << n) - 1
        for i in range(1, n):  # Don't include start vertex
            if dp[full_mask][i] and self.adjacency_matrix[i][0]:
                return True
        
        return False
    
    def find_hamiltonian_path_backtrack(self) -> Optional[List[int]]:
        """
        Approach 4: Find Actual Hamiltonian Path
        
        Return the actual path if one exists.
        
        Time: O(n!), Space: O(n)
        """
        if not self.vertices:
            return []
        
        n = len(self.vertices)
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()
        
        def backtrack(path: List[int], visited: Set[int]) -> Optional[List[int]]:
            if len(path) == n:
                return path[:]
            
            current = path[-1] if path else -1
            
            for next_vertex in range(n):
                if next_vertex in visited:
                    continue
                
                if current == -1 or self.adjacency_matrix[current][next_vertex]:
                    path.append(next_vertex)
                    visited.add(next_vertex)
                    
                    result = backtrack(path, visited)
                    if result:
                        return result
                    
                    path.pop()
                    visited.remove(next_vertex)
            
            return None
        
        # Try starting from each vertex
        for start in range(n):
            result = backtrack([start], {start})
            if result:
                # Convert back to original vertex labels
                return [self.idx_to_vertex[i] for i in result]
        
        return None
    
    def has_hamiltonian_path_heuristic(self) -> bool:
        """
        Approach 5: Heuristic-Based Detection
        
        Use graph properties to quickly determine likelihood.
        
        Time: O(n^2), Space: O(n)
        """
        if not self.vertices:
            return True
        
        n = len(self.vertices)
        
        # Necessary conditions for Hamiltonian path
        
        # 1. Graph must be connected (for undirected)
        if not self._is_connected():
            return False
        
        # 2. Degree conditions (Dirac's theorem for cycles)
        degrees = [len(self.graph[v]) for v in self.vertices]
        
        # For Hamiltonian path: at most 2 vertices can have degree 1
        degree_one_count = sum(1 for d in degrees if d == 1)
        if degree_one_count > 2:
            return False
        
        # 3. Ore's theorem (sufficient condition for cycles)
        vertex_list = list(self.vertices)
        for i in range(n):
            for j in range(i + 1, n):
                u, v = vertex_list[i], vertex_list[j]
                if v not in self.graph[u]:  # Non-adjacent vertices
                    if len(self.graph[u]) + len(self.graph[v]) < n - 1:
                        # This doesn't guarantee no Hamiltonian path, but suggests it's less likely
                        pass
        
        # If passes basic tests, assume likely to have Hamiltonian path
        return True
    
    def count_hamiltonian_paths_dp(self) -> int:
        """
        Approach 6: Count All Hamiltonian Paths
        
        Count the number of distinct Hamiltonian paths.
        
        Time: O(n^2 * 2^n), Space: O(n * 2^n)
        """
        if not self.vertices:
            return 1
        
        n = len(self.vertices)
        if self.adjacency_matrix is None:
            self.build_adjacency_matrix()
        
        # dp[mask][i] = number of ways to visit nodes in mask, ending at node i
        dp = [[0] * n for _ in range(1 << n)]
        
        # Initialize: single nodes
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if dp[mask][u] == 0:
                    continue
                
                for v in range(n):
                    if (mask & (1 << v)) == 0 and self.adjacency_matrix[u][v]:
                        new_mask = mask | (1 << v)
                        dp[new_mask][v] += dp[mask][u]
        
        # Sum all ways to visit all nodes
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n))
    
    def _is_connected(self) -> bool:
        """Check if graph is connected"""
        if not self.vertices:
            return True
        
        start = next(iter(self.vertices))
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(neighbor for neighbor in self.graph[node] 
                           if neighbor not in visited)
        
        return len(visited) == len(self.vertices)

def test_hamiltonian_detection():
    """Test Hamiltonian path detection algorithms"""
    print("=== Testing Hamiltonian Path Detection ===")
    
    # Test cases: (edges, directed, expected_path, expected_cycle, description)
    test_cases = [
        # Simple cases
        ([(0, 1), (1, 2), (2, 3)], False, True, False, "Path graph"),
        ([(0, 1), (1, 2), (2, 3), (3, 0)], False, True, True, "Cycle graph"),
        ([(0, 1), (0, 2), (1, 2)], False, True, True, "Triangle"),
        
        # Complete graphs
        ([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)], False, True, True, "K4"),
        
        # Disconnected
        ([(0, 1), (2, 3)], False, False, False, "Disconnected"),
        
        # Edge cases
        ([], False, True, True, "Empty graph"),
        ([(0, 1)], False, True, False, "Single edge"),
    ]
    
    for edges, directed, expected_path, expected_cycle, description in test_cases:
        print(f"\n--- {description} ---")
        
        graph = HamiltonianPathDetection()
        for u, v in edges:
            graph.add_edge(u, v, directed)
        
        print(f"Edges: {edges}")
        
        # Test path detection
        algorithms = [
            ("Backtrack", graph.has_hamiltonian_path_backtrack),
            ("DP Bitmask", graph.has_hamiltonian_path_dp_bitmask),
            ("Heuristic", graph.has_hamiltonian_path_heuristic),
        ]
        
        print("Hamiltonian Path:")
        for alg_name, alg_func in algorithms:
            try:
                result = alg_func()
                status = "✓" if result == expected_path else "✗"
                print(f"  {alg_name:12} | {status} | Has Path: {result}")
            except Exception as e:
                print(f"  {alg_name:12} | ERROR: {str(e)[:30]}")
        
        # Test cycle detection
        try:
            cycle_result = graph.has_hamiltonian_cycle_dp_bitmask()
            cycle_status = "✓" if cycle_result == expected_cycle else "✗"
            print(f"Hamiltonian Cycle: {cycle_status} | Has Cycle: {cycle_result}")
        except Exception as e:
            print(f"Hamiltonian Cycle: ERROR: {str(e)[:30]}")
        
        # Find actual path
        try:
            path = graph.find_hamiltonian_path_backtrack()
            if path:
                print(f"Path found: {path}")
            else:
                print("No path found")
        except Exception as e:
            print(f"Path finding ERROR: {str(e)[:30]}")
        
        # Count paths
        try:
            if len(graph.vertices) <= 6:  # Only for small graphs
                count = graph.count_hamiltonian_paths_dp()
                print(f"Total Hamiltonian paths: {count}")
        except Exception as e:
            print(f"Path counting ERROR: {str(e)[:30]}")

def demonstrate_hamiltonian_concepts():
    """Demonstrate Hamiltonian path concepts"""
    print("\n=== Hamiltonian Path Concepts ===")
    
    print("Definitions:")
    print("• Hamiltonian Path: Visits each vertex exactly once")
    print("• Hamiltonian Cycle: Hamiltonian path returning to start")
    print("• NP-Complete: No known polynomial algorithm")
    print("• Different from Eulerian (edges vs vertices)")
    
    print("\nKey Properties:")
    print("• Not all graphs have Hamiltonian paths")
    print("• Complete graphs always have Hamiltonian cycles")
    print("• Degree conditions provide necessary conditions")
    print("• Connectivity is necessary but not sufficient")
    
    print("\nAlgorithmic Approaches:")
    print("• Backtracking: Exhaustive search with pruning")
    print("• Dynamic Programming: Bitmask state compression")
    print("• Heuristics: Use graph properties for quick tests")
    print("• Approximation: Near-optimal solutions")
    
    print("\nApplications:")
    print("• Traveling Salesman Problem")
    print("• Circuit board routing")
    print("• DNA sequencing")
    print("• Scheduling problems")

def analyze_hamiltonian_complexity():
    """Analyze complexity of Hamiltonian algorithms"""
    print("\n=== Hamiltonian Algorithm Complexity ===")
    
    print("Algorithm Comparison:")
    
    print("\n1. **Backtracking:**")
    print("   • Time: O(n!) - try all permutations")
    print("   • Space: O(n) - recursion stack")
    print("   • Pros: Simple, finds actual path")
    print("   • Cons: Exponential time")
    
    print("\n2. **DP with Bitmask:**")
    print("   • Time: O(n^2 * 2^n)")
    print("   • Space: O(n * 2^n)")
    print("   • Pros: Better than backtracking")
    print("   • Cons: Exponential space")
    
    print("\n3. **Heuristic Methods:**")
    print("   • Time: O(n^2) - polynomial")
    print("   • Space: O(n)")
    print("   • Pros: Fast, good for filtering")
    print("   • Cons: Not exact, false positives")
    
    print("\n4. **Approximation:**")
    print("   • Various approximation ratios")
    print("   • Polynomial time algorithms")
    print("   • Trade accuracy for speed")
    
    print("\nPractical Considerations:")
    print("• Exact algorithms: n ≤ 20-25")
    print("• Heuristics: Larger graphs")
    print("• Special cases: Polynomial solutions exist")
    print("• Preprocessing: Reduce problem size")

if __name__ == "__main__":
    test_hamiltonian_detection()
    demonstrate_hamiltonian_concepts()
    analyze_hamiltonian_complexity()

"""
Hamiltonian Path Detection demonstrates classic NP-complete problems
and various algorithmic approaches from exact exponential algorithms
to polynomial-time heuristics and approximations.
"""
