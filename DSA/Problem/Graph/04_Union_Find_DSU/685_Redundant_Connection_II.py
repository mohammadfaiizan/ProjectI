"""
685. Redundant Connection II
Difficulty: Hard

Problem:
In this problem, a rooted tree is a directed graph such that, there is exactly one node 
(the root) for which all other nodes are descendants of this node, plus every node has 
exactly at most one parent, except for the root node which has no parent.

The given input is a directed graph that started as a rooted tree with n nodes (with 
distinct values from 1 to n), with one additional directed edge added. The added edge 
has two different vertices chosen from 1 to n, and was not an edge that already existed.

The resulting graph is given as a 2D-array of edges. Each element of edges is a pair 
[ui, vi] that represents a directed edge connecting nodes ui and vi, where ui is a 
parent of vi.

Return an edge that can be removed so that the resulting graph is a rooted tree of n 
nodes. If there are multiple answers, return the answer that occurs last in the input.

Examples:
Input: edges = [[1,2],[1,3],[2,4]]
Output: [2,4]

Input: edges = [[1,2],[2,3],[3,4],[4,1],[1,5]]
Output: [4,1]

Input: edges = [[2,1],[3,1],[4,2],[1,4]]
Output: [2,1]

Constraints:
- n == edges.length
- 3 <= n <= 1000
- edges[i].length == 2
- 1 <= ui, vi <= n
"""

from typing import List

class UnionFind:
    """Union-Find for cycle detection in directed graphs"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank, returns True if union performed"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already connected, would create cycle
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True

class Solution:
    def findRedundantDirectedConnection_approach1_three_cases(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 1: Three-Case Analysis (Optimal)
        
        Handle three cases:
        1. Node with two parents (in-degree 2)
        2. Cycle with no node having two parents
        3. Both cycle and node with two parents
        
        Time: O(N * α(N)) ≈ O(N)
        Space: O(N)
        """
        n = len(edges)
        parent = [0] * (n + 1)  # parent[i] = parent of node i (0 means no parent)
        candidate1 = candidate2 = None
        
        # Step 1: Find node with two parents (if any)
        for u, v in edges:
            if parent[v] == 0:
                parent[v] = u
            else:
                # Found node v with two parents: parent[v] and u
                candidate1 = [parent[v], v]  # First edge to v
                candidate2 = [u, v]          # Second edge to v
                break
        
        # Step 2: Use Union-Find to detect cycles
        uf = UnionFind(n + 1)
        
        for u, v in edges:
            # Skip candidate2 if it exists (try removing second edge first)
            if candidate2 and [u, v] == candidate2:
                continue
            
            if not uf.union(u, v):
                # Found cycle
                if candidate1:
                    # Case 3: Both cycle and two parents
                    # The problematic edge is candidate1 (first edge to the node)
                    return candidate1
                else:
                    # Case 2: Cycle only, no node with two parents
                    return [u, v]
        
        # No cycle found after skipping candidate2
        # Case 1: Node with two parents, no cycle
        return candidate2
    
    def findRedundantDirectedConnection_approach2_dfs_cycle_detection(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 2: DFS-Based Cycle Detection
        
        Use DFS to detect cycles and handle two-parent cases.
        
        Time: O(N^2) - potentially O(N) DFS for each edge
        Space: O(N)
        """
        from collections import defaultdict
        
        def has_cycle_with_removal(skip_edge):
            """Check if graph has cycle when removing skip_edge"""
            graph = defaultdict(list)
            
            for i, (u, v) in enumerate(edges):
                if i != skip_edge:
                    graph[u].append(v)
            
            visited = set()
            rec_stack = set()
            
            def dfs(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph[node]:
                    if dfs(neighbor):
                        return True
                
                rec_stack.remove(node)
                return False
            
            # Check for cycles from all nodes
            for node in range(1, len(edges) + 1):
                if node not in visited:
                    if dfs(node):
                        return True
            
            return False
        
        def is_valid_tree(skip_edge):
            """Check if removing skip_edge creates valid rooted tree"""
            in_degree = [0] * (len(edges) + 1)
            
            for i, (u, v) in enumerate(edges):
                if i != skip_edge:
                    in_degree[v] += 1
            
            # Should have exactly one root (in-degree 0)
            roots = sum(1 for i in range(1, len(edges) + 1) if in_degree[i] == 0)
            
            # All other nodes should have exactly one parent
            valid_parents = all(in_degree[i] <= 1 for i in range(1, len(edges) + 1))
            
            return roots == 1 and valid_parents and not has_cycle_with_removal(skip_edge)
        
        # Try removing each edge from last to first
        for i in range(len(edges) - 1, -1, -1):
            if is_valid_tree(i):
                return edges[i]
        
        return []  # Should never reach here
    
    def findRedundantDirectedConnection_approach3_topological_sort(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 3: Topological Sort with Edge Removal
        
        Use topological sort to validate tree structure.
        
        Time: O(N^2)
        Space: O(N)
        """
        from collections import defaultdict, deque
        
        def can_form_tree(skip_edge):
            """Check if graph forms valid tree when skipping edge"""
            graph = defaultdict(list)
            in_degree = defaultdict(int)
            nodes = set()
            
            for i, (u, v) in enumerate(edges):
                if i != skip_edge:
                    graph[u].append(v)
                    in_degree[v] += 1
                    nodes.add(u)
                    nodes.add(v)
            
            # Check in-degree constraints
            root_count = 0
            for node in nodes:
                if in_degree[node] == 0:
                    root_count += 1
                elif in_degree[node] > 1:
                    return False  # Multiple parents
            
            if root_count != 1:
                return False  # Should have exactly one root
            
            # Topological sort to check for cycles
            queue = deque([node for node in nodes if in_degree[node] == 0])
            processed = 0
            
            while queue:
                node = queue.popleft()
                processed += 1
                
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            return processed == len(nodes)
        
        # Try removing edges from last to first
        for i in range(len(edges) - 1, -1, -1):
            if can_form_tree(i):
                return edges[i]
        
        return []
    
    def findRedundantDirectedConnection_approach4_parent_tracking(self, edges: List[List[int]]) -> List[int]:
        """
        Approach 4: Explicit Parent Tracking
        
        Track parent relationships and detect violations.
        
        Time: O(N)
        Space: O(N)
        """
        n = len(edges)
        parent = {}
        
        # Find edges that create double parent situation
        first_edge = second_edge = None
        
        for u, v in edges:
            if v in parent:
                first_edge = [parent[v], v]
                second_edge = [u, v]
                break
            parent[v] = u
        
        # Helper function to detect cycle using Union-Find
        def has_cycle_without_edge(skip_edge):
            uf = UnionFind(n + 1)
            
            for u, v in edges:
                if [u, v] == skip_edge:
                    continue
                
                if not uf.union(u, v):
                    return True
            
            return False
        
        # Case analysis
        if not second_edge:
            # No double parent, just find cycle-creating edge
            uf = UnionFind(n + 1)
            for u, v in edges:
                if not uf.union(u, v):
                    return [u, v]
        else:
            # Double parent exists
            if has_cycle_without_edge(second_edge):
                # Removing second edge still leaves cycle, so first edge is problematic
                return first_edge
            else:
                # Removing second edge resolves all issues
                return second_edge
        
        return []

def test_redundant_directed_connection():
    """Test all approaches with various cases"""
    solution = Solution()
    
    test_cases = [
        # (edges, expected)
        ([[1,2],[1,3],[2,4]], [2,4]),
        ([[1,2],[2,3],[3,4],[4,1],[1,5]], [4,1]),
        ([[2,1],[3,1],[4,2],[1,4]], [2,1]),
        ([[1,2],[2,3],[3,1]], [3,1]),
        ([[1,2],[1,3],[3,4],[4,1]], [4,1]),
    ]
    
    approaches = [
        ("Three-Case Analysis", solution.findRedundantDirectedConnection_approach1_three_cases),
        ("DFS Cycle Detection", solution.findRedundantDirectedConnection_approach2_dfs_cycle_detection),
        ("Topological Sort", solution.findRedundantDirectedConnection_approach3_topological_sort),
        ("Parent Tracking", solution.findRedundantDirectedConnection_approach4_parent_tracking),
    ]
    
    for approach_name, func in approaches:
        print(f"\n=== {approach_name} Approach ===")
        for i, (edges, expected) in enumerate(test_cases):
            result = func(edges[:])  # Copy to avoid modification
            status = "✓" if result == expected else "✗"
            print(f"Test {i+1}: {status} Edges: {edges}, Expected: {expected}, Got: {result}")

def demonstrate_three_cases():
    """Demonstrate the three cases in directed graph problems"""
    print("\n=== Three Cases in Directed Graph Analysis ===")
    
    cases = [
        {
            "name": "Case 1: Node with Two Parents",
            "edges": [[1,2],[3,2],[2,4]],
            "description": "Node 2 has two parents (1 and 3), no cycle",
            "expected": [3,2]
        },
        {
            "name": "Case 2: Cycle Only", 
            "edges": [[1,2],[2,3],[3,1],[1,4]],
            "description": "Cycle exists (1→2→3→1), no double parent",
            "expected": [3,1]
        },
        {
            "name": "Case 3: Both Cycle and Two Parents",
            "edges": [[2,1],[3,1],[4,2],[1,4]],
            "description": "Node 1 has two parents AND cycle exists",
            "expected": [2,1]
        }
    ]
    
    solution = Solution()
    
    for case in cases:
        print(f"\n{case['name']}:")
        print(f"  Description: {case['description']}")
        print(f"  Edges: {case['edges']}")
        
        # Analyze the case
        edges = case['edges']
        n = len(edges)
        parent = [0] * (n + 1)
        
        # Find double parent
        candidate1 = candidate2 = None
        for u, v in edges:
            if parent[v] == 0:
                parent[v] = u
            else:
                candidate1 = [parent[v], v]
                candidate2 = [u, v]
                print(f"  Double parent detected: Node {v} has parents {parent[v]} and {u}")
                break
        
        if not candidate2:
            print(f"  No double parent found")
        
        # Check for cycles
        uf = UnionFind(n + 1)
        cycle_edge = None
        
        for u, v in edges:
            if candidate2 and [u, v] == candidate2:
                continue
            
            if not uf.union(u, v):
                cycle_edge = [u, v]
                print(f"  Cycle detected with edge: {cycle_edge}")
                break
        
        if not cycle_edge:
            print(f"  No cycle found")
        
        result = solution.findRedundantDirectedConnection_approach1_three_cases(edges[:])
        print(f"  Expected: {case['expected']}")
        print(f"  Result: {result}")

def analyze_directed_vs_undirected():
    """Analyze differences between directed and undirected redundant connection"""
    print("\n=== Directed vs Undirected Graph Analysis ===")
    
    print("Key Differences:")
    
    print("\n1. **Edge Direction Matters:**")
    print("   • Undirected: [1,2] same as [2,1]")
    print("   • Directed: [1,2] means 1→2 (parent→child)")
    
    print("\n2. **Tree Properties:**")
    print("   • Undirected tree: n-1 edges, connected, no cycles")
    print("   • Directed tree (rooted): n-1 edges, one root, each node has ≤1 parent")
    
    print("\n3. **Problem Complexity:**")
    print("   • Undirected: Find any cycle-creating edge")
    print("   • Directed: Handle multiple cases (double parent, cycle, both)")
    
    print("\n4. **Union-Find Usage:**")
    print("   • Undirected: Direct cycle detection")
    print("   • Directed: More complex analysis needed")
    
    print("\nDirected Graph Constraints:")
    print("• Exactly one root (in-degree 0)")
    print("• All other nodes have exactly one parent (in-degree 1)")
    print("• No cycles allowed")
    print("• Tree structure: parent→child relationships")
    
    print("\nThree-Case Analysis:")
    print("• **Case 1:** Node with two parents, no cycle")
    print("  - Remove the later edge to the double-parent node")
    print("• **Case 2:** Cycle exists, no double parent")
    print("  - Remove any edge in the cycle (prefer later one)")
    print("• **Case 3:** Both double parent and cycle")
    print("  - Complex analysis needed to find correct edge")

def demonstrate_union_find_limitations():
    """Demonstrate limitations of Union-Find for directed graphs"""
    print("\n=== Union-Find Limitations in Directed Graphs ===")
    
    print("Why Union-Find Alone Isn't Sufficient:")
    
    print("\n1. **Direction Ignorance:**")
    print("   • Union-Find treats edges as undirected")
    print("   • Can't detect 'parent→child' violations")
    print("   • Misses in-degree constraint violations")
    
    print("\n2. **Parent Relationship:**")
    print("   • Directed tree: each node has ≤1 parent")
    print("   • Union-Find can't track parent relationships")
    print("   • Need separate parent tracking")
    
    print("\n3. **Root Detection:**")
    print("   • Rooted tree needs exactly one root")
    print("   • Union-Find doesn't identify roots")
    print("   • Need in-degree analysis")
    
    print("\nHybrid Solution Strategy:")
    print("• **Step 1:** Detect nodes with multiple parents")
    print("• **Step 2:** Use Union-Find for cycle detection")
    print("• **Step 3:** Combine results with case analysis")
    print("• **Step 4:** Apply removal strategy based on case")
    
    example = [[2,1],[3,1],[4,2],[1,4]]
    print(f"\nExample: {example}")
    print("• Node 1 has parents: 2, 3 (violates single parent)")
    print("• Cycle: 2→1→4→2 (violates acyclic property)")
    print("• Union-Find detects cycle but misses parent violation")
    print("• Need combined analysis to find correct edge")

def compare_solution_strategies():
    """Compare different solution strategies"""
    print("\n=== Solution Strategy Comparison ===")
    
    print("1. **Three-Case Analysis (Optimal):**")
    print("   ✅ O(N) time complexity")
    print("   ✅ Handles all cases systematically")
    print("   ✅ Clear case separation")
    print("   ✅ Union-Find for efficient cycle detection")
    print("   ❌ Requires careful case analysis")
    
    print("\n2. **DFS Cycle Detection:**")
    print("   ✅ Intuitive cycle detection")
    print("   ✅ Direct validation of tree property")
    print("   ❌ O(N²) time complexity")
    print("   ❌ Repeated graph reconstruction")
    
    print("\n3. **Topological Sort:**")
    print("   ✅ Standard algorithm for DAG validation")
    print("   ✅ Natural fit for directed graphs")
    print("   ❌ O(N²) time complexity")
    print("   ❌ Overkill for this specific problem")
    
    print("\n4. **Parent Tracking:**")
    print("   ✅ Explicit parent relationship management")
    print("   ✅ Clear double-parent detection")
    print("   ✅ Efficient implementation")
    print("   ❌ Essentially same as three-case analysis")
    
    print("\nWhen to Use Each:")
    print("• **Three-Case Analysis:** Production code, optimal performance")
    print("• **DFS:** Educational purposes, small graphs")
    print("• **Topological Sort:** When part of larger DAG analysis")
    print("• **Parent Tracking:** When explicit parent info needed")
    
    print("\nReal-world Applications:")
    print("• **Organizational Charts:** Prevent reporting loops")
    print("• **Dependency Graphs:** Remove circular dependencies")
    print("• **File Systems:** Maintain directory tree structure")
    print("• **Database Schema:** Foreign key relationship validation")

if __name__ == "__main__":
    test_redundant_directed_connection()
    demonstrate_three_cases()
    analyze_directed_vs_undirected()
    demonstrate_union_find_limitations()
    compare_solution_strategies()

"""
Union-Find Concepts:
1. Hybrid Data Structure Usage
2. Directed Graph Cycle Detection
3. Constraint Violation Analysis
4. Multi-Case Problem Decomposition

Key Problem Insights:
- Directed graphs have stricter constraints than undirected
- Union-Find alone insufficient for directed tree validation
- Need hybrid approach: parent tracking + cycle detection
- Three cases: double parent, cycle, or both

Algorithm Strategy:
1. Detect nodes with multiple parents
2. Use Union-Find for cycle detection
3. Apply case-based analysis
4. Remove appropriate edge based on case

Directed Tree Constraints:
- Exactly one root (in-degree 0)
- All other nodes have one parent (in-degree 1)
- No cycles allowed
- Parent-child relationships must be preserved

Advanced Technique:
- Combine multiple data structures
- Union-Find for efficiency where applicable
- Custom logic for constraint checking
- Systematic case analysis

Real-world Applications:
- Organizational hierarchy validation
- Dependency graph management
- File system integrity
- Database relationship enforcement
- Network topology verification

This problem demonstrates advanced Union-Find usage
in complex constraint satisfaction scenarios.
"""
