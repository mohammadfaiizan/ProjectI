"""
785. Is Graph Bipartite?
Difficulty: Medium (Listed as Easy in syllabus)

Problem:
There is an undirected graph with n nodes, where each node is numbered between 0 and n - 1. 
You are given a 2D array graph, where graph[i] is an array of nodes that are adjacent to node i. 
More formally, for each edge between nodes i and j, graph[i] contains j and graph[j] contains i.

A graph is bipartite if the nodes can be partitioned into two independent sets A and B such that 
every edge in the graph connects a node in set A and a node in set B.

Return true if and only if it is bipartite.

Examples:
Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
Explanation: There is no way to partition the nodes into two independent sets such that every edge connects a node in one and a node in the other.

Input: graph = [[1,3],[0,2],[1,3],[0,2]]
Output: true
Explanation: We can partition the nodes into two sets: {0, 2} and {1, 3}.

Constraints:
- graph.length == n
- 1 <= n <= 100
- 0 <= graph[i].length < n
- 0 <= graph[i][j] <= n - 1
- graph[i] does not contain i.
- All the values of graph[i] are unique.
- The graph is guaranteed to be undirected.
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import deque, defaultdict

class Solution:
    def isBipartite_approach1_bfs_coloring(self, graph: List[List[int]]) -> bool:
        """
        Approach 1: BFS with 2-Coloring
        
        Use BFS to color nodes with two colors. If we can color all nodes
        without conflicts, the graph is bipartite.
        
        Time: O(V + E)
        Space: O(V)
        """
        n = len(graph)
        color = [-1] * n  # -1: unvisited, 0: color A, 1: color B
        
        # Check each connected component
        for start in range(n):
            if color[start] == -1:
                # BFS to color this component
                queue = deque([start])
                color[start] = 0
                
                while queue:
                    node = queue.popleft()
                    current_color = color[node]
                    next_color = 1 - current_color
                    
                    for neighbor in graph[node]:
                        if color[neighbor] == -1:
                            # Color the neighbor with opposite color
                            color[neighbor] = next_color
                            queue.append(neighbor)
                        elif color[neighbor] == current_color:
                            # Conflict: neighbor has same color
                            return False
        
        return True
    
    def isBipartite_approach2_dfs_coloring(self, graph: List[List[int]]) -> bool:
        """
        Approach 2: DFS with 2-Coloring
        
        Use DFS to color nodes with two colors recursively.
        
        Time: O(V + E)
        Space: O(V) for recursion stack
        """
        n = len(graph)
        color = [-1] * n
        
        def dfs(node, node_color):
            """DFS to color current node and its neighbors"""
            color[node] = node_color
            
            for neighbor in graph[node]:
                if color[neighbor] == -1:
                    # Color neighbor with opposite color
                    if not dfs(neighbor, 1 - node_color):
                        return False
                elif color[neighbor] == node_color:
                    # Conflict: neighbor has same color
                    return False
            
            return True
        
        # Check each connected component
        for start in range(n):
            if color[start] == -1:
                if not dfs(start, 0):
                    return False
        
        return True
    
    def isBipartite_approach3_union_find_optimization(self, graph: List[List[int]]) -> bool:
        """
        Approach 3: Union-Find with Bipartite Property
        
        Use Union-Find to track connected components and detect odd cycles.
        
        Time: O(V + E * α(V))
        Space: O(V)
        """
        n = len(graph)
        
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
                    return False
                
                if self.rank[px] < self.rank[py]:
                    px, py = py, px
                
                self.parent[py] = px
                if self.rank[px] == self.rank[py]:
                    self.rank[px] += 1
                
                return True
            
            def connected(self, x, y):
                return self.find(x) == self.find(y)
        
        # Create Union-Find for original nodes and their "opposites"
        # Node i's opposite is node i + n
        uf = UnionFind(2 * n)
        
        for node in range(n):
            for neighbor in graph[node]:
                # Check if node and neighbor are already in same partition
                if uf.connected(node, neighbor):
                    return False
                
                # Union node with neighbor's opposite
                # Union neighbor with node's opposite
                uf.union(node, neighbor + n)
                uf.union(neighbor, node + n)
        
        return True
    
    def isBipartite_approach4_iterative_dfs(self, graph: List[List[int]]) -> bool:
        """
        Approach 4: Iterative DFS with Stack
        
        Avoid recursion depth issues with iterative DFS.
        
        Time: O(V + E)
        Space: O(V)
        """
        n = len(graph)
        color = [-1] * n
        
        for start in range(n):
            if color[start] == -1:
                # Iterative DFS
                stack = [(start, 0)]
                
                while stack:
                    node, node_color = stack.pop()
                    
                    if color[node] != -1:
                        # Already colored, check consistency
                        if color[node] != node_color:
                            return False
                        continue
                    
                    color[node] = node_color
                    
                    for neighbor in graph[node]:
                        if color[neighbor] == -1:
                            stack.append((neighbor, 1 - node_color))
                        elif color[neighbor] == node_color:
                            return False
        
        return True
    
    def isBipartite_approach5_advanced_analysis(self, graph: List[List[int]]) -> bool:
        """
        Approach 5: Advanced Bipartite Analysis with Detailed Tracking
        
        Comprehensive analysis with component tracking and detailed validation.
        
        Time: O(V + E)
        Space: O(V)
        """
        n = len(graph)
        color = [-1] * n
        components = []
        
        def analyze_component(start):
            """Analyze a single connected component"""
            component_nodes = []
            partition_A = []
            partition_B = []
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                component_nodes.append(node)
                
                if color[node] == 0:
                    partition_A.append(node)
                else:
                    partition_B.append(node)
                
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return None  # Not bipartite
            
            return {
                'nodes': component_nodes,
                'partition_A': partition_A,
                'partition_B': partition_B,
                'size': len(component_nodes),
                'edges': sum(len(graph[node]) for node in component_nodes) // 2
            }
        
        # Analyze each connected component
        for start in range(n):
            if color[start] == -1:
                component_info = analyze_component(start)
                if component_info is None:
                    return False
                components.append(component_info)
        
        # Validate bipartite property for each component
        for component in components:
            # Verify no edges within partitions
            for node in component['partition_A']:
                for neighbor in graph[node]:
                    if neighbor in component['partition_A']:
                        return False
            
            for node in component['partition_B']:
                for neighbor in graph[node]:
                    if neighbor in component['partition_B']:
                        return False
        
        return True

def test_is_bipartite():
    """Test all approaches with various test cases"""
    solution = Solution()
    
    test_cases = [
        # (graph, expected)
        ([[1,2,3],[0,2],[0,1,3],[0,2]], False),
        ([[1,3],[0,2],[1,3],[0,2]], True),
        ([[]], True),
        ([[1],[0]], True),
        ([[1,2],[0,2],[0,1]], False),  # Triangle
        ([[2,4],[2,3,4],[0,1],[1],[0,1]], True),
        ([[3],[2,4],[1],[0],[1]], True),
    ]
    
    approaches = [
        ("BFS Coloring", solution.isBipartite_approach1_bfs_coloring),
        ("DFS Coloring", solution.isBipartite_approach2_dfs_coloring),
        ("Union-Find", solution.isBipartite_approach3_union_find_optimization),
        ("Iterative DFS", solution.isBipartite_approach4_iterative_dfs),
        ("Advanced Analysis", solution.isBipartite_approach5_advanced_analysis),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (graph, expected) in enumerate(test_cases):
            result = func([neighbors[:] for neighbors in graph])  # Deep copy
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} expected={expected}, got={result}")

def demonstrate_bipartite_detection():
    """Demonstrate bipartite detection with visual examples"""
    print("\n=== Bipartite Detection Demo ===")
    
    # Example 1: Bipartite graph
    bipartite_graph = [[1,3],[0,2],[1,3],[0,2]]
    
    print(f"Example 1 - Bipartite Graph:")
    print(f"Graph: {bipartite_graph}")
    print(f"Structure:")
    print(f"  0 -- 1")
    print(f"  |    |")
    print(f"  3 -- 2")
    print(f"Partitions: {{0, 2}} and {{1, 3}}")
    
    solution = Solution()
    result1 = solution.isBipartite_approach1_bfs_coloring(bipartite_graph)
    print(f"Is bipartite: {result1}")
    
    # Example 2: Non-bipartite graph
    print(f"\nExample 2 - Non-Bipartite Graph:")
    non_bipartite_graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
    print(f"Graph: {non_bipartite_graph}")
    print(f"Structure:")
    print(f"  0 -- 1")
    print(f"  |\\   |")
    print(f"  | \\ |")
    print(f"  3 -- 2")
    print(f"Contains triangle: 0-1-2, so not bipartite")
    
    result2 = solution.isBipartite_approach1_bfs_coloring(non_bipartite_graph)
    print(f"Is bipartite: {result2}")

def demonstrate_coloring_algorithm():
    """Demonstrate the coloring algorithm step by step"""
    print("\n=== Coloring Algorithm Demo ===")
    
    graph = [[1,3],[0,2],[1,3],[0,2]]
    n = len(graph)
    color = [-1] * n
    
    print(f"Graph: {graph}")
    print(f"Starting BFS coloring from node 0:")
    
    queue = deque([0])
    color[0] = 0
    step = 1
    
    print(f"Step {step}: Color node 0 with color 0")
    print(f"  Colors: {color}")
    step += 1
    
    while queue:
        node = queue.popleft()
        current_color = color[node]
        next_color = 1 - current_color
        
        print(f"Step {step}: Processing node {node} (color {current_color})")
        
        for neighbor in graph[node]:
            if color[neighbor] == -1:
                color[neighbor] = next_color
                queue.append(neighbor)
                print(f"  Color neighbor {neighbor} with color {next_color}")
            elif color[neighbor] == current_color:
                print(f"  Conflict! Neighbor {neighbor} has same color {current_color}")
                break
        
        print(f"  Colors: {color}")
        step += 1
    
    print(f"\nFinal coloring:")
    print(f"  Partition A (color 0): {[i for i in range(n) if color[i] == 0]}")
    print(f"  Partition B (color 1): {[i for i in range(n) if color[i] == 1]}")

def analyze_bipartite_properties():
    """Analyze mathematical properties of bipartite graphs"""
    print("\n=== Bipartite Graph Properties ===")
    
    print("Mathematical Properties:")
    
    print("\n1. **Definition:**")
    print("   • Graph whose vertices can be divided into two disjoint sets")
    print("   • Every edge connects a vertex from one set to the other")
    print("   • No edges within the same set")
    print("   • Also called 2-colorable graphs")
    
    print("\n2. **Characterization:**")
    print("   • A graph is bipartite ⟺ it contains no odd-length cycles")
    print("   • Equivalent to 2-coloring problem")
    print("   • Can be detected in O(V + E) time")
    print("   • Connected components can be checked independently")
    
    print("\n3. **Detection Algorithm:**")
    print("   • Use BFS/DFS with 2-coloring")
    print("   • Start with arbitrary color for unvisited nodes")
    print("   • Color neighbors with opposite color")
    print("   • If conflict arises, graph is not bipartite")
    
    print("\n4. **Applications:**")
    print("   • Matching problems (jobs to workers)")
    print("   • Scheduling (tasks to time slots)")
    print("   • Resource allocation")
    print("   • Network flow modeling")
    
    print("\n5. **Advanced Properties:**")
    print("   • Maximum matching in bipartite graphs: O(VE)")
    print("   • Perfect matching exists ⟺ Hall's condition")
    print("   • Minimum vertex cover = Maximum matching (König's theorem)")
    print("   • Bipartite graphs are perfect graphs")

def demonstrate_real_world_applications():
    """Demonstrate real-world applications of bipartite graphs"""
    print("\n=== Real-World Applications ===")
    
    print("Bipartite Graph Applications:")
    
    print("\n1. **Job Assignment:**")
    print("   • Workers ↔ Jobs")
    print("   • Each worker can do certain jobs")
    print("   • Find maximum assignment")
    print("   • Example: 3 workers, 4 jobs")
    
    # Simulate job assignment
    job_graph = {
        'Worker1': ['Job1', 'Job3'],
        'Worker2': ['Job2', 'Job4'],
        'Worker3': ['Job1', 'Job4']
    }
    
    print(f"   Assignment graph: {job_graph}")
    print(f"   This forms a bipartite graph")
    
    print("\n2. **Course Scheduling:**")
    print("   • Students ↔ Time Slots")
    print("   • Each student has available times")
    print("   • Schedule courses without conflicts")
    print("   • Bipartite matching finds optimal schedule")
    
    print("\n3. **Recommendation Systems:**")
    print("   • Users ↔ Items")
    print("   • Connections represent preferences")
    print("   • Collaborative filtering")
    print("   • Find similar users/items")
    
    print("\n4. **Network Analysis:**")
    print("   • Social networks (users ↔ groups)")
    print("   • Web graphs (pages ↔ topics)")
    print("   • Biological networks (genes ↔ diseases)")
    print("   • Communication networks (senders ↔ receivers)")
    
    print("\n5. **Game Theory:**")
    print("   • Two-player games")
    print("   • Strategy matching")
    print("   • Auction mechanisms")
    print("   • Resource competition")

def compare_detection_algorithms():
    """Compare different bipartite detection algorithms"""
    print("\n=== Algorithm Comparison ===")
    
    print("Bipartite Detection Algorithms:")
    
    print("\n1. **BFS Approach:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V)")
    print("   • Level-by-level processing")
    print("   • Good for finding shortest paths")
    
    print("\n2. **DFS Approach:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V) recursion stack")
    print("   • Depth-first exploration")
    print("   • Natural recursive implementation")
    
    print("\n3. **Union-Find Approach:**")
    print("   • Time: O(V + E × α(V))")
    print("   • Space: O(V)")
    print("   • Uses complement graph concept")
    print("   • Good for dynamic connectivity")
    
    print("\n4. **Iterative DFS:**")
    print("   • Time: O(V + E)")
    print("   • Space: O(V)")
    print("   • Avoids recursion depth issues")
    print("   • Explicit stack management")
    
    print("\nSelection Guidelines:")
    print("• **Simple graphs:** BFS or DFS")
    print("• **Deep graphs:** Iterative DFS")
    print("• **Dynamic updates:** Union-Find")
    print("• **Shortest paths needed:** BFS")

def demonstrate_edge_cases():
    """Demonstrate edge cases and special scenarios"""
    print("\n=== Edge Cases and Special Scenarios ===")
    
    solution = Solution()
    
    print("Special Cases:")
    
    # Empty graph
    print("\n1. **Empty Graph:**")
    empty_graph = [[]]
    result = solution.isBipartite_approach1_bfs_coloring(empty_graph)
    print(f"   Graph: {empty_graph}")
    print(f"   Is bipartite: {result} (trivially true)")
    
    # Single edge
    print("\n2. **Single Edge:**")
    single_edge = [[1], [0]]
    result = solution.isBipartite_approach1_bfs_coloring(single_edge)
    print(f"   Graph: {single_edge}")
    print(f"   Is bipartite: {result}")
    
    # Isolated vertices
    print("\n3. **Isolated Vertices:**")
    isolated = [[], [], []]
    result = solution.isBipartite_approach1_bfs_coloring(isolated)
    print(f"   Graph: {isolated}")
    print(f"   Is bipartite: {result} (isolated vertices)")
    
    # Star graph
    print("\n4. **Star Graph:**")
    star = [[1,2,3,4], [0], [0], [0], [0]]
    result = solution.isBipartite_approach1_bfs_coloring(star)
    print(f"   Graph: {star}")
    print(f"   Is bipartite: {result} (center vs leaves)")
    
    # Complete bipartite graph K₂,₃
    print("\n5. **Complete Bipartite K₂,₃:**")
    k23 = [[2,3,4], [2,3,4], [0,1], [0,1], [0,1]]
    result = solution.isBipartite_approach1_bfs_coloring(k23)
    print(f"   Graph: {k23}")
    print(f"   Is bipartite: {result}")

if __name__ == "__main__":
    test_is_bipartite()
    demonstrate_bipartite_detection()
    demonstrate_coloring_algorithm()
    analyze_bipartite_properties()
    demonstrate_real_world_applications()
    compare_detection_algorithms()
    demonstrate_edge_cases()

"""
Bipartite Graph Detection Concepts:
1. 2-Coloring Algorithm using BFS and DFS
2. Graph Partitioning and Independent Set Analysis
3. Odd Cycle Detection and Characterization Theorem
4. Union-Find Applications in Bipartite Verification
5. Real-world Modeling with Bipartite Structures

Key Problem Insights:
- Bipartite ⟺ No odd cycles ⟺ 2-colorable
- BFS/DFS coloring is optimal O(V + E) approach
- Connected components can be checked independently
- Applications in matching, scheduling, and resource allocation

Algorithm Strategy:
1. Use BFS/DFS to assign colors to vertices
2. Color neighbors with opposite colors
3. Detect conflicts (same color neighbors)
4. Handle multiple connected components

Graph Theory Foundation:
- Bipartite graphs are fundamental in matching theory
- König's theorem connects matching to vertex cover
- Hall's theorem characterizes perfect matchings
- Applications span from algorithms to social networks

Optimization Techniques:
- Early termination on conflict detection
- Component-wise processing for efficiency
- Union-Find for dynamic connectivity queries
- Iterative approaches to avoid stack overflow

Real-world Applications:
- Job assignment and workforce optimization
- Course scheduling and resource allocation
- Recommendation systems and collaborative filtering
- Network analysis and social graph modeling
- Game theory and strategic interactions

This problem serves as foundation for advanced bipartite
algorithms including maximum matching and flow networks.
"""
