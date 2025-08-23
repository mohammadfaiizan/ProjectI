"""
Contest Graph Basics - Competitive Programming Fundamentals
Difficulty: Easy

This file covers essential graph algorithms and techniques commonly used
in competitive programming contests. Focus on implementation speed,
optimization tricks, and contest-specific considerations.

Key Concepts:
1. Fast I/O and Graph Representation
2. Common Graph Algorithms (Optimized)
3. Contest-Specific Optimizations
4. Template Code and Utilities
5. Time/Space Complexity Optimization
6. Edge Case Handling
"""

from typing import List, Dict, Set, Tuple, Optional, Deque
from collections import defaultdict, deque
import sys
from io import StringIO

class ContestGraphBasics:
    """Essential graph algorithms for competitive programming"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
        self.n = 0
        self.m = 0
    
    def fast_input_parsing(self, input_str: str) -> Tuple[int, int, List[Tuple[int, int]]]:
        """
        Approach 1: Fast Input Parsing
        
        Optimized input parsing for contest environments.
        
        Time: O(M), Space: O(M)
        """
        lines = input_str.strip().split('\n')
        n, m = map(int, lines[0].split())
        
        edges = []
        for i in range(1, m + 1):
            u, v = map(int, lines[i].split())
            edges.append((u, v))
        
        return n, m, edges
    
    def build_adjacency_list_fast(self, n: int, edges: List[Tuple[int, int]], 
                                 directed: bool = False) -> List[List[int]]:
        """
        Approach 2: Fast Adjacency List Construction
        
        Optimized for contest environments with pre-allocated lists.
        
        Time: O(V + E), Space: O(V + E)
        """
        # Pre-allocate lists for better performance
        adj = [[] for _ in range(n + 1)]
        
        for u, v in edges:
            adj[u].append(v)
            if not directed:
                adj[v].append(u)
        
        return adj
    
    def dfs_iterative_fast(self, adj: List[List[int]], start: int, n: int) -> List[bool]:
        """
        Approach 3: Fast Iterative DFS
        
        Optimized DFS avoiding recursion limits.
        
        Time: O(V + E), Space: O(V)
        """
        visited = [False] * (n + 1)
        stack = [start]
        
        while stack:
            node = stack.pop()
            
            if visited[node]:
                continue
                
            visited[node] = True
            
            # Add neighbors in reverse order for consistent traversal
            for neighbor in reversed(adj[node]):
                if not visited[neighbor]:
                    stack.append(neighbor)
        
        return visited
    
    def bfs_fast(self, adj: List[List[int]], start: int, n: int) -> Tuple[List[int], List[int]]:
        """
        Approach 4: Fast BFS with Distance and Parent Tracking
        
        Optimized BFS for shortest path and tree construction.
        
        Time: O(V + E), Space: O(V)
        """
        dist = [-1] * (n + 1)
        parent = [-1] * (n + 1)
        queue = deque([start])
        
        dist[start] = 0
        
        while queue:
            node = queue.popleft()
            
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        return dist, parent
    
    def connected_components_fast(self, adj: List[List[int]], n: int) -> Tuple[int, List[int]]:
        """
        Approach 5: Fast Connected Components
        
        Find connected components using optimized DFS.
        
        Time: O(V + E), Space: O(V)
        """
        visited = [False] * (n + 1)
        component = [0] * (n + 1)
        comp_count = 0
        
        for i in range(1, n + 1):
            if not visited[i]:
                comp_count += 1
                
                # DFS to mark component
                stack = [i]
                while stack:
                    node = stack.pop()
                    
                    if visited[node]:
                        continue
                    
                    visited[node] = True
                    component[node] = comp_count
                    
                    for neighbor in adj[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
        
        return comp_count, component
    
    def cycle_detection_undirected_fast(self, adj: List[List[int]], n: int) -> bool:
        """
        Approach 6: Fast Cycle Detection (Undirected)
        
        Optimized cycle detection for undirected graphs.
        
        Time: O(V + E), Space: O(V)
        """
        visited = [False] * (n + 1)
        
        def dfs(node, parent):
            visited[node] = True
            
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:
                    return True
            
            return False
        
        for i in range(1, n + 1):
            if not visited[i]:
                if dfs(i, -1):
                    return True
        
        return False
    
    def bipartite_check_fast(self, adj: List[List[int]], n: int) -> Tuple[bool, List[int]]:
        """
        Approach 7: Fast Bipartite Graph Check
        
        Check if graph is bipartite using BFS coloring.
        
        Time: O(V + E), Space: O(V)
        """
        color = [-1] * (n + 1)
        
        for start in range(1, n + 1):
            if color[start] == -1:
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    node = queue.popleft()
                    
                    for neighbor in adj[node]:
                        if color[neighbor] == -1:
                            color[neighbor] = 1 - color[node]
                            queue.append(neighbor)
                        elif color[neighbor] == color[node]:
                            return False, []
        
        return True, color
    
    def shortest_path_unweighted_fast(self, adj: List[List[int]], start: int, 
                                    end: int, n: int) -> Tuple[int, List[int]]:
        """
        Approach 8: Fast Shortest Path (Unweighted)
        
        Find shortest path in unweighted graph with path reconstruction.
        
        Time: O(V + E), Space: O(V)
        """
        if start == end:
            return 0, [start]
        
        dist = [-1] * (n + 1)
        parent = [-1] * (n + 1)
        queue = deque([start])
        
        dist[start] = 0
        
        while queue:
            node = queue.popleft()
            
            if node == end:
                break
            
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
        
        if dist[end] == -1:
            return -1, []
        
        # Reconstruct path
        path = []
        current = end
        while current != -1:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return dist[end], path
    
    def tree_diameter_fast(self, adj: List[List[int]], n: int) -> Tuple[int, List[int]]:
        """
        Approach 9: Fast Tree Diameter (Two BFS)
        
        Find diameter of tree using two BFS calls.
        
        Time: O(V + E), Space: O(V)
        """
        # First BFS from arbitrary node to find one end
        dist1, _ = self.bfs_fast(adj, 1, n)
        farthest1 = max(range(1, n + 1), key=lambda x: dist1[x])
        
        # Second BFS from farthest node to find diameter
        dist2, parent = self.bfs_fast(adj, farthest1, n)
        farthest2 = max(range(1, n + 1), key=lambda x: dist2[x])
        
        diameter = dist2[farthest2]
        
        # Reconstruct diameter path
        path = []
        current = farthest2
        while current != -1:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        return diameter, path
    
    def contest_template_functions(self):
        """
        Approach 10: Contest Template Functions
        
        Collection of utility functions for contests.
        """
        
        def read_ints():
            return list(map(int, input().split()))
        
        def read_int():
            return int(input())
        
        def print_list(arr):
            print(' '.join(map(str, arr)))
        
        def yes_no(condition):
            print("YES" if condition else "NO")
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        # Fast modular arithmetic
        MOD = 10**9 + 7
        
        def mod_add(a, b):
            return (a + b) % MOD
        
        def mod_mul(a, b):
            return (a * b) % MOD
        
        def mod_pow(base, exp, mod=MOD):
            result = 1
            while exp > 0:
                if exp % 2 == 1:
                    result = (result * base) % mod
                base = (base * base) % mod
                exp //= 2
            return result
        
        return {
            'read_ints': read_ints,
            'read_int': read_int,
            'print_list': print_list,
            'yes_no': yes_no,
            'gcd': gcd,
            'lcm': lcm,
            'mod_add': mod_add,
            'mod_mul': mod_mul,
            'mod_pow': mod_pow
        }

def test_contest_basics():
    """Test contest graph basics"""
    print("=== Testing Contest Graph Basics ===")
    
    # Sample input
    input_str = """5 6
1 2
2 3
3 4
4 5
5 1
2 4"""
    
    solver = ContestGraphBasics()
    
    # Test input parsing
    n, m, edges = solver.fast_input_parsing(input_str)
    print(f"Parsed: n={n}, m={m}, edges={edges}")
    
    # Build adjacency list
    adj = solver.build_adjacency_list_fast(n, edges)
    print(f"Adjacency list built for {n} vertices, {m} edges")
    
    # Test algorithms
    print(f"\n--- Algorithm Tests ---")
    
    # DFS
    visited = solver.dfs_iterative_fast(adj, 1, n)
    print(f"DFS from 1: visited = {[i for i in range(1, n+1) if visited[i]]}")
    
    # BFS
    dist, parent = solver.bfs_fast(adj, 1, n)
    print(f"BFS from 1: distances = {dist[1:n+1]}")
    
    # Connected components
    comp_count, components = solver.connected_components_fast(adj, n)
    print(f"Connected components: {comp_count}")
    
    # Cycle detection
    has_cycle = solver.cycle_detection_undirected_fast(adj, n)
    print(f"Has cycle: {has_cycle}")
    
    # Bipartite check
    is_bipartite, colors = solver.bipartite_check_fast(adj, n)
    print(f"Is bipartite: {is_bipartite}")
    
    # Shortest path
    dist_path, path = solver.shortest_path_unweighted_fast(adj, 1, 4, n)
    print(f"Shortest path 1->4: distance={dist_path}, path={path}")

def demonstrate_contest_optimization():
    """Demonstrate contest optimization techniques"""
    print("\n=== Contest Optimization Techniques ===")
    
    print("Fast I/O Techniques:")
    print("• Use sys.stdin.readline() for faster input")
    print("• Pre-parse all input at once")
    print("• Use list comprehensions for bulk operations")
    print("• Avoid string concatenation in loops")
    
    print("\nGraph Representation:")
    print("• Adjacency list: Best for sparse graphs")
    print("• Pre-allocate lists: [[] for _ in range(n+1)]")
    print("• Use 1-indexed arrays for easier mapping")
    print("• Consider edge list for specific algorithms")
    
    print("\nAlgorithm Optimizations:")
    print("• Iterative DFS: Avoid recursion limits")
    print("• Early termination: Stop when answer found")
    print("• Bit manipulation: Use for state compression")
    print("• Modular arithmetic: Handle large numbers")
    
    print("\nMemory Optimizations:")
    print("• Reuse arrays when possible")
    print("• Use appropriate data types (int vs long)")
    print("• Clear collections after use if needed")
    print("• Consider space-time tradeoffs")

def analyze_contest_complexity():
    """Analyze complexity considerations for contests"""
    print("\n=== Contest Complexity Analysis ===")
    
    print("Time Complexity Guidelines:")
    print("• O(1): ~10^8 operations per second")
    print("• O(log n): Very fast, use liberally")
    print("• O(n): Good for n ≤ 10^6")
    print("• O(n log n): Good for n ≤ 10^5")
    print("• O(n²): Good for n ≤ 10^3")
    print("• O(n³): Good for n ≤ 500")
    
    print("\nSpace Complexity Guidelines:")
    print("• Usually 256MB memory limit")
    print("• int array of size 10^6: ~4MB")
    print("• Adjacency list: O(V + E) space")
    print("• 2D arrays: Consider memory carefully")
    
    print("\nCommon Contest Constraints:")
    print("• 1 ≤ n ≤ 10^5 (most graph problems)")
    print("• 1 ≤ m ≤ 2×10^5 (edge count)")
    print("• Time limit: 1-2 seconds typically")
    print("• Multiple test cases: Factor into complexity")
    
    print("\nOptimization Priorities:")
    print("1. Correctness first")
    print("2. Time complexity optimization")
    print("3. Implementation speed")
    print("4. Space optimization (if needed)")
    print("5. Code readability (for debugging)")

if __name__ == "__main__":
    test_contest_basics()
    demonstrate_contest_optimization()
    analyze_contest_complexity()

"""
Contest Graph Basics - Key Insights:

1. **Contest Environment:**
   - Fast I/O is crucial for large inputs
   - Implementation speed matters
   - Template code saves time
   - Edge cases must be handled

2. **Graph Representation:**
   - Adjacency lists for most problems
   - 1-indexed arrays for easier mapping
   - Pre-allocated structures for performance
   - Consider memory constraints

3. **Algorithm Selection:**
   - Choose simplest correct algorithm
   - Iterative over recursive when possible
   - Early termination for optimization
   - Standard algorithms with optimizations

4. **Implementation Tips:**
   - Use proven template code
   - Handle edge cases systematically
   - Test with sample inputs
   - Consider integer overflow

5. **Contest Strategy:**
   - Read all problems first
   - Solve easier problems quickly
   - Use standard algorithms
   - Debug systematically

This foundation provides the essential tools
for competitive programming graph problems.
"""
